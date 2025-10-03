import hashlib
import logging
from typing import List, Dict, Tuple
from tqdm import tqdm
from . import cleaning, llm_scorer, chunker

def cleaning_pipeline(documents: List[Dict[str, str]], config: Dict) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Runs the full text cleaning, chunking, filtering, and optional LLM scoring pipeline.
    """
    logging.info("=== Starting Cleaning, Chunking & Filtering Pipeline ===\n")
    
    seen_hashes = set()
    final_documents = []
    discarded_documents = []
    
    filter_config = config['filters']
    cleaning_config = config['cleaning']
    
    logging.info("=== Stage 1/5: Initial Text Cleaning ===")
    for doc in tqdm(documents, desc="Cleaning text"):
        text = doc.get('text', '')
        title = doc.get('title', '')
        text = cleaning.remove_wiki_markup(text)
        text = cleaning.remove_repeated_title(text, title)
        text = cleaning.remove_tags(text)
        if cleaning_config['use_aggressive_char_removal']:
            text = cleaning.strict_mal_char(text)
        text = cleaning.remove_extra_whitespace(text)
        doc['text'] = text
    
    logging.info(f"✓ Cleaned {len(documents)} documents\n")
    
    logging.info("=== Stage 2/5: Semantic Chunking ===")
    chunked_documents = chunker.chunk_documents(documents, config)
    
    logging.info("=== Stage 3/5: Pre-filtering Chunks ===")
    logging.info("Filtering chunks by Malayalam ratio and word count...")
    
    chunks_to_score = []
    for chunk in tqdm(chunked_documents, desc="Filtering chunks"):
        text = chunk['text']
        passes_malayalam_filter = cleaning.filter_by_ratio(
            text, 
            threshold=filter_config['malayalam_ratio_threshold']
        )
        passes_word_count_filter = cleaning.filter_by_count(
            text, 
            min_words=filter_config['min_word_count']
        )

        if passes_malayalam_filter and passes_word_count_filter:
            chunks_to_score.append(chunk)
        else:
            reason = []
            if not passes_malayalam_filter: 
                reason.append("failed malayalam ratio")
            if not passes_word_count_filter: 
                reason.append("failed word count")
            
            chunk['discard_reason'] = ", ".join(reason)
            chunk['discard_stage'] = "Stage 3: Pre-filtering"
            discarded_documents.append(chunk)
    
    total_chunks = len(chunked_documents)
    passed_chunks = len(chunks_to_score)
    discarded_chunks = len(discarded_documents)
    pass_rate = (passed_chunks / total_chunks * 100) if total_chunks > 0 else 0
    
    logging.info(f"✓ Passed: {passed_chunks}/{total_chunks} chunks ({pass_rate:.1f}%)")
    logging.info(f"✗ Discarded: {discarded_chunks} chunks ({100-pass_rate:.1f}%)\n")
    
    scored_documents = []
    if config.get('llm_scorer', {}).get('enabled', False):
        logging.info("=== Stage 4/5: LLM Scoring ===")
        logging.info(f"Scoring {len(chunks_to_score)} chunks that passed pre-filtering...")
        scored_documents = llm_scorer.score_documents(chunks_to_score, config)
        logging.info("")
    else:
        logging.info("=== Stage 4/5: LLM Scoring ===")
        logging.info("LLM scoring disabled. Skipping...\n")
        scored_documents = chunks_to_score
        for doc in scored_documents:
            if 'llm_score' not in doc:
                doc['llm_score'] = None
                doc['llm_reason'] = 'LLM scoring disabled'

    logging.info("=== Stage 5/5: Final Filtering & Deduplication ===")
    scorer_config = config.get('llm_scorer', {})
    score_threshold = scorer_config.get('score_threshold', 5)

    for doc in tqdm(scored_documents, desc="Final filtering"):
        text = doc['text']
        llm_score = doc.get('llm_score')

        if llm_score is not None and llm_score < score_threshold:
            doc['discard_reason'] = f"failed llm score ({llm_score})"
            doc['discard_stage'] = "Stage 5: LLM score filtering"
            discarded_documents.append(doc)
            continue

        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            final_documents.append(doc)
        else:
            doc['discard_reason'] = "duplicate content"
            doc['discard_stage'] = "Stage 5: Deduplication"
            discarded_documents.append(doc)
    
    logging.info("=== Pipeline Complete ===")
    
    total_processed = len(final_documents) + len(discarded_documents)
    success_rate = (len(final_documents) / total_processed * 100) if total_processed > 0 else 0
    
    logging.info(f"Total chunks processed: {total_processed}")
    logging.info(f"✓ Final dataset: {len(final_documents)} chunks ({success_rate:.1f}%)")
    logging.info(f"✗ Discarded: {len(discarded_documents)} chunks ({100-success_rate:.1f}%)")
    
    if discarded_documents:
        logging.info("Discard breakdown:")
        discard_reasons = {}
        for doc in discarded_documents:
            reason = doc.get('discard_reason', 'unknown')
            discard_reasons[reason] = discard_reasons.get(reason, 0) + 1
        
        for reason, count in sorted(discard_reasons.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"  - {reason}: {count} chunks")
    
    logging.info("")
    
    return final_documents, discarded_documents