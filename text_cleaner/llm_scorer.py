import json
import logging
import os
import re
import time
from typing import List, Dict
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import google.generativeai as genai
from tqdm import tqdm


def _prompt(text: str) -> str:
    """
    Creates the prompt for the LLM to score the text.
    """
    return f"""You are a Linguistic Quality Assurance Specialist for Malayalam. Your task is to provide a critical and precise evaluation of the following text. Your primary goal is to identify and fail text that is semantically or logically incoherent. You must respond ONLY with a valid JSON object.

Follow this thought process step-by-step:
1.  **Fluency Analysis:** Read the text. Does it flow naturally in Malayalam?
2.  **Coherence Analysis:** This is the most important step. Do the sentences and paragraphs logically connect? Is it sensical?
3.  **Final Scoring:** Based on your analysis, assign a score from 1 to 10. Prioritize coherence above all else.

SCORING RUBRIC:
- **Score 8-10 (High Quality):** Fluent, fully coherent, and well-structured.
- **Score 6-7 (Acceptable Quality):** Generally fluent and coherent, may have minor stylistic issues.
- **Score 4-5 (Low Quality):** Difficult to read, contains grammatical errors OR a noticeable lack of logical coherence.
- **Score 1-3 (Garbage Quality):** Completely unacceptable, grammatically broken, semantically nonsensical, or not Malayalam.

Analyze the following text and provide your score and a brief, precise reason in the specified JSON format.

Text:
\"\"\"
{text}
\"\"\"

JSON Response:
"""


def _score_with_api(documents: List[Dict[str, str]], config: Dict) -> List[Dict[str, str]]:
    """
    Scores pre-chunked documents using the Google Gemini API with retry logic.
    
    Args:
        documents: List of document chunks (already chunked, one chunk per document)
        config: API configuration dictionary
    
    Returns:
        Same documents list with llm_score and llm_reason fields added
    """
    api_config = config['api_config']
    max_retries = api_config.get('max_retries', 3)
    retry_delay = api_config.get('retry_delay_seconds', 5)
    
    logging.info(f"Initializing Gemini API: {api_config['model_name']}")

    load_dotenv()
    
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        generation_config = genai.GenerationConfig(response_mime_type="application/json")
        model = genai.GenerativeModel(api_config['model_name'], generation_config=generation_config)
    except Exception as e:
        logging.error(f"Failed to initialize Gemini API: {e}")
        for doc in documents:
            doc['llm_score'] = 1
            doc['llm_reason'] = f"API initialization failed: {str(e)}"
        return documents

    logging.info("API initialized. Starting chunk scoring...")
    
    for doc in tqdm(documents, desc="Scoring chunks (API)"):
        if not doc.get('text', '').strip():
            doc['llm_score'] = 1
            doc['llm_reason'] = "Empty text content"
            continue
        
        prompt = _prompt(doc['text'])
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                response_json = json.loads(response.text)
                doc['llm_score'] = int(response_json.get('score', 1))
                doc['llm_reason'] = response_json.get('reason', 'No reason provided by API.')
                break
                
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(
                    f"Attempt {attempt + 1}/{max_retries}: Invalid JSON from API "
                    f"for {doc.get('doc_name', 'unknown')} chunk {doc.get('chunk_id', 'N/A')}. "
                    f"Error: {e}"
                )
                doc['llm_score'] = 1
                doc['llm_reason'] = "Invalid JSON response from API"
                break
                
            except Exception as e:
                logging.error(
                    f"Attempt {attempt + 1}/{max_retries}: API error "
                    f"for {doc.get('doc_name', 'unknown')} chunk {doc.get('chunk_id', 'N/A')}: {e}"
                )
                if attempt + 1 == max_retries:
                    doc['llm_score'] = 1
                    doc['llm_reason'] = 'API error after multiple retries'
                else:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

    return documents


def _score_with_local(documents: List[Dict[str, str]], config: Dict) -> List[Dict[str, str]]:
    """
    Scores pre-chunked documents using a local GGUF model.
    
    Args:
        documents: List of document chunks (already chunked, one chunk per document)
        config: Local model configuration dictionary
    
    Returns:
        Same documents list with llm_score and llm_reason fields added
    """
    local_config = config['local_config']
    logging.info(f"Initializing local model: {local_config['model_repo_id']}")
    
    try:
        model_path = hf_hub_download(
            repo_id=local_config['model_repo_id'], 
            filename=local_config['model_filename']
        )
        context_size = local_config.get('n_ctx', 8192)
        llm = Llama(
            model_path=model_path, 
            n_ctx=context_size, 
            n_gpu_layers=local_config.get('n_gpu_layers', -1),
            verbose=False
        )
    except Exception as e:
        logging.error(f"Failed to load local model: {e}")
        for doc in documents:
            doc['llm_score'] = 1
            doc['llm_reason'] = f"Model loading failed: {str(e)}"
        return documents

    logging.info("Local model loaded. Starting chunk scoring...")

    for doc in tqdm(documents, desc="Scoring chunks (Local)"):
        if not doc.get('text', '').strip():
            doc['llm_score'] = 1
            doc['llm_reason'] = "Empty text content"
            continue
        
        prompt = _prompt(doc['text'])
        
        try:
            response = llm(prompt, max_tokens=512, temperature=0.0)
            raw_output = response['choices'][0]['text']
            
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                response_json = json.loads(json_string)
                doc['llm_score'] = int(response_json.get('score', 1))
                doc['llm_reason'] = response_json.get('reason', 'No reason provided.')
            else:
                logging.warning(
                    f"No JSON found in model output for {doc.get('doc_name', 'unknown')} "
                    f"chunk {doc.get('chunk_id', 'N/A')}. "
                    f"Raw output: '{raw_output[:100]}...'"
                )
                doc['llm_score'] = 1
                doc['llm_reason'] = "No valid JSON in model output"
        
        except json.JSONDecodeError as e:
            logging.warning(
                f"Invalid JSON from local model for {doc.get('doc_name', 'unknown')} "
                f"chunk {doc.get('chunk_id', 'N/A')}: '{json_string[:100]}...'. "
                f"Error: {e}"
            )
            doc['llm_score'] = 1
            doc['llm_reason'] = f"Invalid JSON: {json_string[:50]}"
        
        except Exception as e:
            logging.error(
                f"Inference error for {doc.get('doc_name', 'unknown')} "
                f"chunk {doc.get('chunk_id', 'N/A')}: {e}"
            )
            doc['llm_score'] = 1
            doc['llm_reason'] = 'Inference error'
    
    return documents


def score_documents(documents: List[Dict[str, str]], config: Dict) -> List[Dict[str, str]]:
    """
    Score documents using configured LLM provider.
    
    Args:
        documents: List of pre-chunked documents with metadata
        config: Full configuration dictionary
    
    Returns:
        Documents with llm_score and llm_reason fields added
    """
    scorer_config = config.get('llm_scorer', {})
    provider = scorer_config.get('provider')
    
    if not documents:
        logging.info("No documents to score.")
        return []

    logging.info(f"LLM Scorer: Using provider '{provider}'")
    
    if provider == 'api':
        return _score_with_api(documents, scorer_config)
    elif provider == 'local':
        return _score_with_local(documents, scorer_config)
    else:
        logging.warning(f"Unknown provider '{provider}'. Skipping LLM scoring.")
        for doc in documents:
            doc['llm_score'] = None
            doc['llm_reason'] = f'Unknown provider: {provider}'
        return documents