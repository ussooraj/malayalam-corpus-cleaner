import json
from pathlib import Path
from text_cleaner.ingestion import ingest_from_directory
from text_cleaner.pipeline import run_cleaning_pipeline
from text_cleaner.formatting import save_to_jsonl

RAW_DATA_PATH = "data/test"
PROCESSED_DATA_PATH = "data/processed"
OUTPUT_FILENAME = "cleaned_malayalam_corpus.jsonl"

def main():
    """
    Main function to run the corpus cleaning pipeline.
    """
    print("--- Starting Corpus Cleaning Toolkit ---")

    documents = ingest_from_directory(RAW_DATA_PATH)
    print(f"Found {len(documents)} documents to process.")

    if not documents:
        print("No documents were found. Exiting.")
        return

    cleaned_documents = run_cleaning_pipeline(documents)

    output_file_path = Path(PROCESSED_DATA_PATH) / OUTPUT_FILENAME
    save_to_jsonl(cleaned_documents, output_file_path)

    print(f"\n--- Verification: Inspecting first 3 of {len(cleaned_documents)} final documents ---")
    for doc in cleaned_documents[:3]:
        print(json.dumps(doc, indent=2, ensure_ascii=False))
        print("-" * 20)

    print("\n--- Toolkit execution finished ---")


if __name__ == "__main__":
    main()