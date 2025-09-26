import docx
import markdown
import logging
from pathlib import Path
from typing import List, Dict, Union
from bs4 import BeautifulSoup

def _extract_docs(text: str, file_path: Path) -> List[Dict[str, str]]:
    """
    Extracts documents from a given text content.
    
    - If <doc> tags are present, it extracts each one as a document.
    - If no <doc> tags are found, it treats the entire text as a single document,
      using the file's name for metadata.
    """
    soup = BeautifulSoup(text, 'html.parser')
    docs_in_text = soup.find_all('doc')
    extracted_documents = []

    if docs_in_text:
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
    else:
        full_text = soup.get_text(strip=True)
        if full_text:
            extracted_documents.append({
                'id': file_path.name,
                'url': str(file_path.resolve()),
                'title': file_path.stem,
                'text': full_text
            })
            
    return extracted_documents

def _handle_text_file(file_path: Path) -> List[Dict[str, str]]:
    """
    Handles .txt, .md, and extensionless files.
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        if file_path.suffix == '.md':
            html = markdown.markdown(content)
            content = BeautifulSoup(html, 'html.parser').get_text()
            
        return _extract_docs(content, file_path)
    except UnicodeDecodeError:
        logging.warning(f"Could not read {file_path.name} as text. Skipping.")
        return []

def _handle_docx_file(file_path: Path) -> List[Dict[str, str]]:
    """
    Handles .docx files by extracting text from paragraphs.
    """
    try:
        document = docx.Document(file_path)
        full_text = "\n".join([para.text for para in document.paragraphs])
        return _extract_docs(full_text, file_path)
    except Exception as e:
        logging.error(f"Could not process DOCX file {file_path.name}: {e}. Skipping.")
        return []

def ingest_from_dir(directory_path: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Ingests and parses all supported files from a directory, handling both
    structured (<doc>) and unstructured (plain text) files.
    """
    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        logging.error(f"Directory not found at {directory_path}")
        return []

    logging.info(f"Starting ingestion from: {directory_path}")
    
    all_documents = []
    
    for file_path in directory_path.iterdir():
        if not file_path.is_file():
            continue

        logging.info(f"  -> Processing file: {file_path.name}")
        
        if file_path.suffix in ['.txt', '.md', '']:
            all_documents.extend(_handle_text_file(file_path))
        elif file_path.suffix == '.docx':
            all_documents.extend(_handle_docx_file(file_path))
        else:
            logging.info(f"Skipping unsupported file type: {file_path.name}")

    logging.info(f"Ingestion complete. Found {len(all_documents)} documents in total.")
    return all_documents