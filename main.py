import argparse
import yaml
import json
import logging
from pathlib import Path
from typing import Dict
from text_cleaner.ingestion import ingest_from_dir
from text_cleaner.pipeline import cleaning_pipeline
from text_cleaner.formatting import save_to_jsonl

def setup_logging():
    """
    Configures the logging for the entire application.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    file_handler = logging.FileHandler('processing.log', mode='w')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_config(config_path: str) -> Dict:
    """
    Loads the YAML configuration file from the given path.
    """
    logging.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        exit()

def main():
    """
    Main function to run the corpus cleaning cleaning_pipeline.
    """
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Corpus Cleaning Toolkit")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    path_config = config['paths']
    
    raw_data_path = path_config['raw_data_dir']
    processed_data_path = path_config['processed_data_dir']
    output_filename = path_config['output_filename']
    
    logging.info("--- Starting Corpus Cleaning Toolkit ---")

    documents = ingest_from_dir(raw_data_path)
    logging.info(f"Found {len(documents)} documents to process.")

    if not documents:
        logging.warning("No documents were found. Exiting.")
        return

    cleaned_documents = cleaning_pipeline(documents, config)

    output_file_path = Path(processed_data_path) / output_filename
    save_to_jsonl(cleaned_documents, output_file_path)

    logging.info(f"--- Verification: Inspecting first 3 of {len(cleaned_documents)} final documents ---")
    for doc in cleaned_documents[:3]:
        print(json.dumps(doc, indent=2, ensure_ascii=False))
        print("-" * 20)

    logging.info("--- Toolkit execution finished ---")

if __name__ == "__main__":
    main()