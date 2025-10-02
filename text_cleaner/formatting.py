import json
import logging
from pathlib import Path
from typing import List, Dict, Union

def save_to_jsonl(documents: List[Dict[str, str]], output_path: Union[str, Path]):
    """
    Saves a list of documents to a JSON Lines (.jsonl) file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving data to: {output_path}")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                json_record = json.dumps(doc, ensure_ascii=False)
                f.write(json_record + '\n')
        
        logging.info(f"Successfully saved {len(documents)} documents.\n")
    
    except Exception as e:
        logging.error(f"Could not save file. {e}")