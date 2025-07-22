"""
Test embedding integration with document ingestion.
"""

import asyncio
import os
from app.services.embedding_service import generate_document_embedding
from app.dto.document_dto import DocumentData
from app.types.document_types import DocumentSource
from dotenv import load_dotenv

load_dotenv()


async def test_embedding_service():
    """Test the embedding service with a sample document."""
    
    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set, skipping embedding test")
        return
    
    # Create a sample document
    sample_doc = DocumentData(
        title="Test Document",
        content="This is a sample document for testing embedding generation. It contains some text that should be converted to a vector representation.",
        source=DocumentSource.CUSTOM,
        original_id="test-123"
    )
    
    try:
        # Generate embedding
        print("🔄 Generating embedding for sample document...")
        embedding = await generate_document_embedding(sample_doc)
        
        # Verify embedding
        if embedding and len(embedding) > 0:
            print(f"✅ Embedding generated successfully!")
            print(f"   - Vector dimensions: {len(embedding)}")
            print(f"   - First few values: {embedding[:5]}")
            print(f"   - Document: {sample_doc.title}")
        else:
            print("❌ Empty embedding generated")
            
    except Exception as e:
        print(f"❌ Failed to generate embedding: {e}")

if __name__ == "__main__":
    asyncio.run(test_embedding_service())