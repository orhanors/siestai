#!/usr/bin/env python3
"""
Test similarity search with different thresholds.
"""

import asyncio
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.append('/Users/orhanors/Desktop/siestai-full/siestai')

from app.memory.history.history import search_similar_messages
from app.services.embedding_service import get_embeddings

async def test_similarity_thresholds():
    """Test different similarity thresholds."""
    print("🔍 Testing similarity search with different thresholds...")
    
    # Initialize embeddings
    embeddings = get_embeddings()
    
    # Test query
    test_query = "What is artificial intelligence and machine learning?"
    test_embedding = await embeddings.aembed_query(test_query)
    
    # Test different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for threshold in thresholds:
        print(f"\n📊 Testing threshold: {threshold}")
        
        # Search for debug_user (should find messages)
        similar_debug = await search_similar_messages(
            embedding=test_embedding,
            user_id="debug_user",
            profile_id="debug_profile",
            limit=5,
            threshold=threshold
        )
        
        print(f"   debug_user found: {len(similar_debug)} messages")
        for msg in similar_debug:
            print(f"     • {msg['content'][:40]}... (similarity: {msg['similarity']:.3f})")
        
        # Search for test_user (might find messages)
        similar_test = await search_similar_messages(
            embedding=test_embedding,
            user_id="test_user_123",
            profile_id="work_profile",
            limit=5,
            threshold=threshold
        )
        
        print(f"   test_user_123 found: {len(similar_test)} messages")
        for msg in similar_test:
            print(f"     • {msg['content'][:40]}... (similarity: {msg['similarity']:.3f})")

if __name__ == "__main__":
    try:
        asyncio.run(test_similarity_thresholds())
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()