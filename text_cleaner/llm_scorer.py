import json
import logging
import os
import re
import time
from typing import List, Dict
from dotenv import load_dotenv
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from indicnlp import common
from indicnlp.tokenize import sentence_tokenize
import google.generativeai as genai
from tqdm import tqdm

INDIC_NLP_RESOURCES_DIR = "indic_nlp_resources"
common.set_resources_path(INDIC_NLP_RESOURCES_DIR)


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

def _semantic_chunking(text: str, tokenizer, target_size: int) -> List[str]:
    chunks = []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    current_chunk_paragraphs = []
    current_chunk_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = len(tokenizer.encode(paragraph))
        
        if paragraph_tokens > target_size:
            if current_chunk_paragraphs: 
                chunks.append("\n\n".join(current_chunk_paragraphs))
            current_chunk_paragraphs, current_chunk_tokens = [], 0
            sentences = sentence_tokenize.sentence_split(paragraph, lang='ml')
            current_sentence_chunk, current_sentence_tokens = [], 0
            
            for sentence in sentences:
                sentence_tokens = len(tokenizer.encode(sentence))
                if current_sentence_tokens + sentence_tokens > target_size:
                    if current_sentence_chunk: 
                        chunks.append(" ".join(current_sentence_chunk))
                    current_sentence_chunk, current_sentence_tokens = [sentence], sentence_tokens
                else:
                    current_sentence_chunk.append(sentence)
                    current_sentence_tokens += sentence_tokens
            if current_sentence_chunk: 
                chunks.append(" ".join(current_sentence_chunk))
            continue
        
        if current_chunk_tokens + paragraph_tokens > target_size:
            if current_chunk_paragraphs: 
                chunks.append("\n\n".join(current_chunk_paragraphs))
            current_chunk_paragraphs, current_chunk_tokens = [paragraph], paragraph_tokens
        else:
            current_chunk_paragraphs.append(paragraph)
            current_chunk_tokens += paragraph_tokens
    
    if current_chunk_paragraphs: 
        chunks.append("\n\n".join(current_chunk_paragraphs))
    return chunks

def _score_with_api(documents: List[Dict[str, str]], config: Dict) -> List[Dict[str, str]]:
    """
    Scores documents using the Google Gemini API with retry logic.
    """
    api_config = config['api_config']

    max_retries = api_config.get('max_retries', 3)
    retry_delay = api_config.get('retry_delay_seconds', 5)
    
    logging.info(f"LLM Scorer: Initializing API with model {api_config['model_name']}")

    load_dotenv()
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        generation_config = genai.GenerationConfig(response_mime_type="application/json")
        model = genai.GenerativeModel(api_config['model_name'], generation_config=generation_config)
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    except Exception as e:
        logging.error(f"Failed to initialize Gemini API. Error: {e}")
        return documents

    logging.info("LLM Scorer: Gemini API client initialized. Starting document scoring.")
    
    for doc in tqdm(documents, desc="Scoring documents (API)"):
        if not doc['text'].strip():
            doc['llm_score'], doc['llm_reason'] = 1, "Empty text content"; continue
            
        chunks = _semantic_chunking(doc['text'], tokenizer, config['chunk_target_size'])
        chunk_results = []
        
        for chunk in chunks:
            prompt = _prompt(chunk)
            response_json = None

            for attempt in range(max_retries):
                try:
                    response = model.generate_content(prompt)
                    response_json = json.loads(response.text)
                    score = int(response_json.get('score', 1))
                    reason = response_json.get('reason', 'No reason provided by API.')
                    chunk_results.append({'score': score, 'reason': reason})
                    break
                except (json.JSONDecodeError, ValueError) as e:
                    logging.warning(f"  -> Attempt {attempt + 1}/{max_retries}: Could not parse JSON from API. Error: {e}")
                    chunk_results.append({'score': 1, 'reason': "Invalid JSON response from API."})
                    break
                except Exception as e:
                    logging.error(f"  -> Attempt {attempt + 1}/{max_retries}: API error: {e}. Retrying in {retry_delay}s...")
                    if attempt + 1 == max_retries:
                        chunk_results.append({'score': 1, 'reason': 'API inference error after multiple retries.'})
                    time.sleep(retry_delay)

        if chunk_results:
            min_result = min(chunk_results, key=lambda x: x['score'])
            doc['llm_score'], doc['llm_reason'] = min_result['score'], min_result['reason']
        else:
            doc['llm_score'], doc['llm_reason'] = 1, "No chunks were processed."
            
    return documents

def _score_with_local(documents: List[Dict[str, str]], config: Dict) -> List[Dict[str, str]]:
    local_config = config['local_config']
    logging.info(f"LLM Scorer: Initializing local model {local_config['model_repo_id']}")
    try:
        model_path = hf_hub_download(repo_id=local_config['model_repo_id'], filename=local_config['model_filename'])
        context_size = local_config.get('n_ctx', 8192)
        llm = Llama(model_path=model_path, n_ctx=context_size, n_gpu_layers=local_config['n_gpu_layers'], verbose=False)
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    except Exception as e:
        logging.error(f"Failed to load GGUF model. Error: {e}"); return documents

    logging.info("LLM Scorer: Model loaded. Starting document scoring.")

    for doc in tqdm(documents, desc="Scoring documents (Local)"):
        if not doc['text'].strip():
            doc['llm_score'], doc['llm_reason'] = 1, "Empty text content"; continue
        
        chunks = _semantic_chunking(doc['text'], tokenizer, config['chunk_target_size'])
        chunk_results = []
        
        for chunk in chunks:
            prompt = _prompt(chunk)
            try:
                response = llm(prompt, max_tokens=512, temperature=0.0)
                raw_output = response['choices'][0]['text']
                json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
                if json_match:
                    json_string = json_match.group(0)
                    response_json = json.loads(json_string)
                    score = int(response_json.get('score', 1))
                    reason = response_json.get('reason', 'No reason provided.')
                    chunk_results.append({'score': score, 'reason': reason})
                else:
                    logging.warning(f"No JSON found in model output for chunk. Raw output (first 100 chars): '{raw_output[:100]}'")
                    chunk_results.append({'score': 1, 'reason': "No valid JSON found in model output"})
            
            except json.JSONDecodeError as e:
                logging.warning(f"Invalid JSON response from model: '{json_string[:100]}'. Error: {e}")
                chunk_results.append({'score': 1, 'reason': f"Invalid JSON response: {json_string[:100]}"})
            except Exception as e:
                logging.error(f"  -> An unexpected error during model inference: {e}")
                chunk_results.append({'score': 1, 'reason': 'Inference error.'})

        if chunk_results:
            min_result = min(chunk_results, key=lambda x: x['score'])
            doc['llm_score'], doc['llm_reason'] = min_result['score'], min_result['reason']
        else:
            doc['llm_score'], doc['llm_reason'] = 1, "No chunks were processed."
        
    return documents

def score_documents(documents: List[Dict[str, str]], config: Dict) -> List[Dict[str, str]]:
    scorer_config = config.get('llm_scorer', {})
    provider = scorer_config.get('provider')
    
    if not documents:
        logging.info("No documents to score.")
        return []

    logging.info(f"LLM Scorer starting with provider: '{provider}'")
    if provider == 'api':
        return _score_with_api(documents, scorer_config)
    elif provider == 'local':
        return _score_with_local(documents, scorer_config)
    else:
        logging.warning(f"LLM Scorer: Unknown provider '{provider}'. Skipping scoring.")
        return documents