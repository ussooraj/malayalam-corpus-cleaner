import json
from pathlib import Path
from typing import List, Dict, Union

def save_to_jsonl(documents: List[Dict[str, str]], output_path: Union[str, Path]):
    """
    Saves a list of cleaned documents to a JSON Lines (.jsonl) file.

    Each dictionary in the list is converted to a JSON string and written
    as a new line in the output file.

    Args:
        documents: The list of cleaned document dictionaries.
        output_path: The full path for the output .jsonl file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Saving cleaned data to {output_path} ---")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                # ensure_ascii=False, for saving Malayalam characters.
                json_record = json.dumps(doc, ensure_ascii=False)
                f.write(json_record + '\n')
        
        print(f"Successfully saved {len(documents)} documents.")
    
    except Exception as e:
        print(f"Error: Could not save file. {e}")