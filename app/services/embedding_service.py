"""
Minimal embedding service for document ingestion.
"""

import os
from typing import List
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils.logger import get_logger
from app.dto.document_dto import DocumentData

logger = get_logger("siestai.embedding")

# Global embedding service instance
_embeddings = None

def get_embeddings():
    """Get OpenAI embeddings instance."""
    global _embeddings
    if _embeddings is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        _embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-3-small"
        )
        logger.info("OpenAI embeddings initialized")
    return _embeddings

async def generate_document_embedding(document: DocumentData) -> List[float]:
    """Generate embedding for a document using chunking for long texts."""
    embeddings = get_embeddings()
    text = f"{document.title}\n\n{document.content}"
    
    # Use LangChain text splitter for chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    if len(chunks) == 1:
        embedding = await embeddings.aembed_query(text)
        logger.debug(f"Generated single embedding for: {document.title}")
        return embedding
    
    # Generate embeddings for each chunk
    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        chunk_embedding = await embeddings.aembed_query(chunk)
        chunk_embeddings.append(chunk_embedding)
        logger.debug(f"Generated embedding for chunk {i+1}/{len(chunks)} of: {document.title}")
    
    # Average the embeddings
    averaged_embedding = np.mean(chunk_embeddings, axis=0).tolist()
    logger.info(f"Generated averaged embedding from {len(chunks)} chunks for: {document.title}")
    return averaged_embedding