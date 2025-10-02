import hashlib
import logging
from typing import List, Dict, Tuple
from tqdm import tqdm
from . import cleaning, llm_scorer

def cleaning_pipeline(documents: List[Dict[str, str]], config: Dict) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Runs the full text cleaning, filtering, and optional LLM scoring pipeline.
    This version includes pre-filtering to optimize performance and cost.
    """
    logging.info("--- Starting Cleaning, Filtering & Deduplication Pipeline ---")
    
    seen_hashes = set()
    final_documents = []
    discarded_documents = []
    
    filter_config = config['filters']
    cleaning_config = config['cleaning']
    
    logging.info("Stage 1: Performing initial text cleaning...")
    for doc in tqdm(documents, desc="Initial Cleaning"):
        text = doc.get('text', '')
        title = doc.get('title', '')
        text = cleaning.remove_wiki_markup(text)
        text = cleaning.remove_repeated_title(text, title)
        text = cleaning.remove_tags(text)
        if cleaning_config['use_aggressive_char_removal']:
            text = cleaning.strict_mal_char(text)
        text = cleaning.remove_extra_whitespace(text)
        doc['text'] = text

    logging.info("Stage 2: Pre-filtering documents by word count and language ratio...")
    documents_to_score = []
    for doc in tqdm(documents, desc="Pre-filtering"):
        text = doc['text']
        passes_malayalam_filter = cleaning.filter_by_ratio(text, threshold=filter_config['malayalam_ratio_threshold'])
        passes_word_count_filter = cleaning.filter_by_count(text, min_words=filter_config['min_word_count'])

        if passes_malayalam_filter and passes_word_count_filter:
            documents_to_score.append(doc)
        else:
            reason = []
            if not passes_malayalam_filter: reason.append("failed malayalam ratio")
            if not passes_word_count_filter: reason.append("failed word count")
            doc['discard_reason'] = ", ".join(reason)
            discarded_documents.append(doc)
    
    logging.info(f"Pre-filtering complete.")
    logging.info(f"{len(documents_to_score)} documents passed")
    logging.info(f"{len(discarded_documents)} documents discarded.")

    scored_documents = []
    if config.get('llm_scorer', {}).get('enabled', False):
        logging.info("Stage 3: LLM scoring is enabled. Starting scoring process...")
        scored_documents = llm_scorer.score_documents(documents_to_score, config)
    else:
        logging.info("Stage 3: LLM scoring is disabled. Skipping.")
        scored_documents = documents_to_score

    logging.info("Stage 4: Performing final filtering and deduplication...\n")
    scorer_config = config.get('llm_scorer', {})
    score_threshold = scorer_config.get('score_threshold', 5)

    for doc in tqdm(scored_documents, desc="Final Filtering"):
        text = doc['text']
        llm_score = doc.get('llm_score')

        if llm_score is not None and llm_score < score_threshold:
            doc['discard_reason'] = f"failed llm score ({llm_score})"
            discarded_documents.append(doc)
            continue

        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            final_documents.append(doc)
        else:
            doc['discard_reason'] = "duplicate content"
            discarded_documents.append(doc)
            
    logging.info("--- Pipeline Complete ---")
    
    total = len(final_documents) + len(discarded_documents)
    success_rate = (len(final_documents) / total * 100) if total > 0 else 0
    
    logging.info(f"Total documents processed: {total}")
    logging.info(f"Cleaned documents: {len(final_documents)} ({success_rate:.1f}%)")
    logging.info(f"Discarded documents: {len(discarded_documents)} ({100-success_rate:.1f}%)\n")
    
    return final_documents, discarded_documents