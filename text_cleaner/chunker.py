import logging
from typing import List, Dict
from indicnlp import common
from indicnlp.tokenize import sentence_tokenize

INDIC_NLP_RESOURCES_DIR = "indic_nlp_resources"
common.set_resources_path(INDIC_NLP_RESOURCES_DIR)


def _semantic_chunking(text: str, target_size: int) -> List[str]:
    """
    Chunks text into semantically coherent segments based on word count.
    Respects paragraph and sentence boundaries to maintain context.
    """
    chunks = []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    current_chunk_paragraphs = []
    current_chunk_words = 0

    for paragraph in paragraphs:
        paragraph_words = len(paragraph.split())
        
        if paragraph_words > target_size:
            if current_chunk_paragraphs: 
                chunks.append("\n\n".join(current_chunk_paragraphs))
                current_chunk_paragraphs, current_chunk_words = [], 0
            
            sentences = sentence_tokenize.sentence_split(paragraph, lang='ml')
            current_sentence_chunk = []
            current_sentence_words = 0
            
            for sentence in sentences:
                sentence_words = len(sentence.split())
                
                if current_sentence_words + sentence_words > target_size:
                    if current_sentence_chunk: 
                        chunks.append(" ".join(current_sentence_chunk))
                    current_sentence_chunk = [sentence]
                    current_sentence_words = sentence_words
                else:
                    current_sentence_chunk.append(sentence)
                    current_sentence_words += sentence_words
            
            if current_sentence_chunk: 
                chunks.append(" ".join(current_sentence_chunk))
            continue
        
        if current_chunk_words + paragraph_words > target_size:
            if current_chunk_paragraphs: 
                chunks.append("\n\n".join(current_chunk_paragraphs))
            current_chunk_paragraphs = [paragraph]
            current_chunk_words = paragraph_words
        else:
            current_chunk_paragraphs.append(paragraph)
            current_chunk_words += paragraph_words
    
    if current_chunk_paragraphs: 
        chunks.append("\n\n".join(current_chunk_paragraphs))
    
    return chunks


def chunk_documents(documents: List[Dict], config: Dict) -> List[Dict]:
    """
    Chunk documents according to config settings and add comprehensive metadata.
    """
    chunking_config = config.get('chunking', {})
    
    if not chunking_config.get('enabled', True):
        logging.info("Chunking disabled, keeping documents as-is")
        total_docs = len(documents)
        doc_id_width = len(str(total_docs))
        
        for doc_index, doc in enumerate(documents, start=1):
            doc['doc_id'] = f"{doc_index:0{doc_id_width}d}"
            doc['doc_name'] = doc.get('filename', f'document_{doc["doc_id"]}')
            doc['doc_source_path'] = doc.get('filepath', '')
            doc['total_chunks'] = 1
            doc['chunk_id'] = 1
        
        return documents
    
    target_size = chunking_config.get('target_size', 5000)
    
    logging.info(f"Semantic chunking enabled with target size: {target_size} words")
    
    chunked_documents = []
    total_docs = len(documents)
    doc_id_width = len(str(total_docs))
    
    for doc_index, doc in enumerate(documents, start=1):
        doc_id = f"{doc_index:0{doc_id_width}d}"

        chunks = _semantic_chunking(doc['text'], target_size)
        
        doc_name = doc.get('filename', f'document_{doc_id}')
        doc_source_path = doc.get('filepath', '')
        
        for chunk_index, chunk_text in enumerate(chunks, start=1):
            chunked_documents.append({
                'doc_id': doc_id,
                'doc_name': doc_name,
                'doc_source_path': doc_source_path,
                'total_chunks': len(chunks),
                'chunk_id': chunk_index,
                'text': chunk_text
            })
        
        total_words = sum(len(chunk.split()) for chunk in chunks)
        avg_words = total_words // len(chunks) if chunks else 0
        
        logging.info(
            f"  ✓ {doc_name}: {len(chunks)} chunks "
            f"(~{avg_words} words/chunk)"
        )
    
    logging.info(
        f"Chunking complete: {total_docs} documents → {len(chunked_documents)} chunks\n"
    )
    
    return chunked_documents