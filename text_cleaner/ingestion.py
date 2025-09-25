import os
import docx
import markdown
from pathlib import Path
from typing import List, Dict, Union
from bs4 import BeautifulSoup

def _parse_doc_struct(text: str) -> List[Dict[str, str]]:
    """A helper function to parse a block of text for <doc> structures."""
    soup = BeautifulSoup(text, 'html.parser')
    docs_in_text = soup.find_all('doc')
    extracted_documents = []
    
    for doc in docs_in_text:
        doc_id = doc.get('id')
        doc_url = doc.get('url')
        doc_title = doc.get('title')
        doc_text = doc.get_text(strip=True)

        if doc_id and doc_title and doc_text:
            extracted_documents.append({
                'id': doc_id,
                'url': doc_url,
                'title': doc_title,
                'text': doc_text
            })
    return extracted_documents

def _handle_text_file(file_path: Path) -> List[Dict[str, str]]:
    """Handles .txt, .md, and extensionless files."""
    try:
        content = file_path.read_text(encoding='utf-8')
        # If it's a markdown file, convert to HTML first to remove markup
        if file_path.suffix == '.md':
            content = markdown.markdown(content)
        return _parse_doc_struct(content)
    except UnicodeDecodeError:
        print(f"  -> WARNING: Could not read {file_path.name} as text. Skipping.")
        return []

def _handle_docx_file(file_path: Path) -> List[Dict[str, str]]:
    """Handles .docx files by extracting text from paragraphs."""
    try:
        document = docx.Document(file_path)
        # Converts all paragraphs into a single string
        full_text = "\n".join([para.text for para in document.paragraphs])
        return _parse_doc_struct(full_text)
    except Exception as e:
        print(f"  -> ERROR: Could not process DOCX file {file_path.name}: {e}. Skipping.")
        return []

def ingest_from_directory(directory_path: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Ingests and parses all supported files from a directory.

    This function iterates through a directory and uses specific handlers
    for different file types (.txt, .md, .docx, and extensionless files)
    to extract <doc>...</doc> structures.

    Args:
        directory_path: The path to the directory containing the raw files.

    Returns:
        A list of dictionaries, one for each document found.
    """
    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        print(f"Error: Directory not found at {directory_path}")
        return []

    print(f"Starting ingestion from: {directory_path}")
    
    all_documents = []
    
    for file_path in directory_path.iterdir():
        if not file_path.is_file():
            continue # Skip subdirectories

        print(f"  -> Processing file: {file_path.name}")
        
        if file_path.suffix in ['.txt', '.md', '']:
            all_documents.extend(_handle_text_file(file_path))
        elif file_path.suffix == '.docx':
            all_documents.extend(_handle_docx_file(file_path))
        else:
            print(f"  -> INFO: Skipping unsupported file type: {file_path.name}")

    print(f"Ingestion complete. Found {len(all_documents)} documents in total.")
    return all_documents