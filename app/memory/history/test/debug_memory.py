#!/usr/bin/env python3
"""
Debug script for memory functionality.
"""

import asyncio
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.append('/Users/orhanors/Desktop/siestai-full/siestai')

from app.memory.history.session_manager import memory_manager
from app.services.embedding_service import get_embeddings

async def debug_memory():
    """Debug memory functionality step by step."""
    print("🔍 Debugging memory functionality...")
    
    # Initialize memory manager
    await memory_manager.initialize()
    print("✅ Memory manager initialized")
    
    # Initialize embeddings
    embeddings = get_embeddings()
    print("✅ Embeddings service initialized")
    
    # Test creating a session
    user_id = "debug_user"
    profile_id = "debug_profile"
    
    session = await memory_manager.get_or_create_session(
        user_id=user_id,
        profile_id=profile_id,
        session_name="Debug Session"
    )
    print(f"✅ Created session: {session.session_id}")
    
    # Test adding messages
    query1 = "What is artificial intelligence?"
    response1 = "Artificial intelligence (AI) is a field of computer science that focuses on creating intelligent machines."
    
    query1_embedding = await embeddings.aembed_query(query1)
    response1_embedding = await embeddings.aembed_query(response1)
    
    await session.add_message(
        role="user",
        content=query1,
        embedding=query1_embedding,
        metadata={"type": "debug_query"}
    )
    print("✅ Added user message 1")
    
    await session.add_message(
        role="assistant", 
        content=response1,
        embedding=response1_embedding,
        metadata={"type": "debug_response"}
    )
    print("✅ Added assistant message 1")
    
    # Add second pair
    query2 = "How does machine learning work?"
    response2 = "Machine learning is a subset of AI that enables computers to learn from data without explicit programming."
    
    query2_embedding = await embeddings.aembed_query(query2)
    response2_embedding = await embeddings.aembed_query(response2)
    
    await session.add_message(
        role="user",
        content=query2,
        embedding=query2_embedding,
        metadata={"type": "debug_query"}
    )
    print("✅ Added user message 2")
    
    await session.add_message(
        role="assistant",
        content=response2,
        embedding=response2_embedding,
        metadata={"type": "debug_response"}
    )
    print("✅ Added assistant message 2")
    
    # Test retrieving memory context
    test_query = "Tell me about AI and ML"
    test_embedding = await embeddings.aembed_query(test_query)
    
    print(f"\n🔍 Testing memory retrieval for query: '{test_query}'")
    
    memory_context = await session.get_memory_context(
        query_embedding=test_embedding,
        max_similar=3,
        max_recent=5
    )
    
    print(f"📊 Memory context retrieved:")
    print(f"   • Current session messages: {len(memory_context.get('current_session', []))}")
    print(f"   • Similar messages: {len(memory_context.get('similar_messages', []))}")
    print(f"   • Recent context: {len(memory_context.get('recent_context', []))}")
    print(f"   • Summaries: {len(memory_context.get('summaries', []))}")
    
    if memory_context.get('similar_messages'):
        print("\n🎯 Similar messages found:")
        for i, msg in enumerate(memory_context['similar_messages']):
            print(f"   {i+1}. Content: {msg['content'][:50]}...")
            print(f"      Similarity: {msg['similarity']:.3f}")
            print(f"      Role: {msg['role']}")
    
    if memory_context.get('current_session'):
        print(f"\n💬 Current session has {len(memory_context['current_session'])} messages")
        for i, msg in enumerate(memory_context['current_session']):
            print(f"   {i+1}. {msg['role']}: {msg['content'][:50]}...")
    
    # Test with different session to check isolation
    print(f"\n🔄 Testing with different session...")
    session2 = await memory_manager.get_or_create_session(
        user_id="different_user",
        profile_id="different_profile",
        session_name="Different Session"
    )
    
    memory_context2 = await session2.get_memory_context(
        query_embedding=test_embedding,
        max_similar=3,
        max_recent=5
    )
    
    print(f"📊 Different user memory context:")
    print(f"   • Similar messages: {len(memory_context2.get('similar_messages', []))}")
    print(f"   • Recent context: {len(memory_context2.get('recent_context', []))}")
    
    # Cleanup
    await session.close_session()
    await session2.close_session()
    
    print("\n🎉 Memory debug completed!")

if __name__ == "__main__":
    try:
        asyncio.run(debug_memory())
    except Exception as e:
        print(f"\n❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()