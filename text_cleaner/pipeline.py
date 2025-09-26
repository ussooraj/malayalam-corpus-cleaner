import hashlib
import logging
from typing import List, Dict, Tuple
from . import cleaning

def cleaning_pipeline(documents: List[Dict[str, str]], config: Dict) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Runs the full text cleaning and filtering pipeline.
    
    Returns:
        A tuple containing two lists: (final_documents, discarded_documents)
    """
    logging.info("--- Starting Cleaning, Filtering & Deduplication Pipeline ---")
    
    seen_hashes = set()
    final_documents = []
    discarded_documents = []
    
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
            reason = []
            if not passes_malayalam_filter:
                reason.append("failed malayalam ratio filter")
            if not passes_word_count_filter:
                reason.append("failed word count filter")

            doc['discard_reason'] = ", ".join(reason)
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
    logging.info(f"Accepted {len(final_documents)} high-quality documents.")
    logging.info(f"Discarded {len(discarded_documents)} documents (due to low quality or duplication).")
    
    return final_documents, discarded_documents