import hashlib
import logging
from typing import List, Dict
from . import cleaning

def cleaning_pipeline(documents: List[Dict[str, str]], config: Dict) -> List[Dict[str, str]]:
    """
    Runs the full text cleaning, filtering, and deduplication pipeline
    using settings from a configuration dictionary.
    """
    logging.info("--- Starting Cleaning, Filtering & Deduplication Pipeline ---")
    
    seen_hashes = set()
    final_documents = []
    initial_doc_count = len(documents)
    duplicates_found = 0

    filter_config = config['filters']
    cleaning_config = config['cleaning']

    for doc in documents:
        text = doc['text']
        title = doc['title']

        text = cleaning.remove_repeated_title(text, title)
        text = cleaning.remove_tags(text)
        
        if cleaning_config['use_aggressive_char_removal']:
            text = cleaning.strict_mal_char(text)
            
        text = cleaning.remove_extra_whitespace(text)
        doc['text'] = text

        passes_malayalam_filter = cleaning.filter_by_ratio(text, threshold=filter_config['malayalam_ratio_threshold'])
        passes_word_count_filter = cleaning.filter_by_count(text, min_words=filter_config['min_word_count'])

        if not (passes_malayalam_filter and passes_word_count_filter):
            continue

        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            final_documents.append(doc)
        else:
            duplicates_found += 1
            
    final_doc_count = len(final_documents)
    low_quality_removed = initial_doc_count - final_doc_count - duplicates_found
    
    logging.info("--- Pipeline Complete ---")
    logging.info(f"Removed {low_quality_removed} low-quality documents.")
    logging.info(f"Removed {duplicates_found} duplicate documents.")
    
    return final_documents