import hashlib
from typing import List, Dict
from . import cleaning

def run_cleaning_pipeline(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Runs the full text cleaning, filtering, and deduplication pipeline.
    """
    print("\n--- Starting Cleaning, Filtering & Deduplication Pipeline ---")
    
    seen_hashes = set()
    
    final_documents = []
    initial_doc_count = len(documents)
    duplicates_found = 0

    for doc in documents:
        text = doc['text']
        title = doc['title']

        text = cleaning.remove_repeated_title(text, title)
        text = cleaning.remove_tags(text)
        text = cleaning.strict_mal_chars(text)
        text = cleaning.remove_extra_whitespace(text)
        
        doc['text'] = text

        passes_malayalam_filter = cleaning.filter_by_ratio(text, threshold=0.8)
        passes_word_count_filter = cleaning.filter_by_word_count(text)

        if not (passes_malayalam_filter and passes_word_count_filter):
            continue

        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest() # For Deduplication

        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            final_documents.append(doc)
        else:
            duplicates_found += 1
            
    final_doc_count = len(final_documents)
    low_quality_removed = initial_doc_count - final_doc_count - duplicates_found
    
    print("--- Pipeline Complete ---")
    print(f"Removed {low_quality_removed} low-quality documents.")
    print(f"Removed {duplicates_found} duplicate documents.")
    
    return final_documents