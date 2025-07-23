#!/usr/bin/env python3
"""
Test script for chat history functionality.
"""

import asyncio
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.append('/Users/orhanors/Desktop/siestai-full/siestai')

from app.agents.research_agent import ResearchAgent

async def test_memory_functionality():
    """Test the memory-enabled research agent."""
    print("🧪 Testing memory-enabled research agent...")
    
    # Test user/profile IDs
    user_id = "test_user_123"
    profile_id = "work_profile"
    
    # Initialize agent with memory
    agent = ResearchAgent(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_documents=3,
        max_web_results=2,
        enable_kg=False,
        enable_web_search=True,
        enable_memory=True
    )
    
    print(f"👤 Testing with User: {user_id}, Profile: {profile_id}")
    
    async with agent:
        # Test 1: First query
        print("\n📝 Test 1: First query about LangGraph")
        result1 = await agent.research(
            query="What is LangGraph?",
            user_id=user_id,
            profile_id=profile_id
        )
        
        print(f"✅ Answer: {result1['answer'][:100]}...")
        print(f"📊 Sources used: {result1['metadata'].get('sources_used', [])}")
        
        # Test 2: Follow-up query (should use memory)
        print("\n📝 Test 2: Follow-up query that should reference memory")
        result2 = await agent.research(
            query="How is it different from LangChain?",
            user_id=user_id,
            profile_id=profile_id
        )
        
        print(f"✅ Answer: {result2['answer'][:100]}...")
        memory_context = result2.get('memory_context', {})
        if memory_context.get('similar_messages'):
            print(f"🧠 Found {len(memory_context['similar_messages'])} similar messages from memory")
        else:
            print("⚠️  No memory context found")
        
        # Test 3: Different user (should not see previous user's memory)
        print("\n📝 Test 3: Different user should not see previous memory")
        result3 = await agent.research(
            query="What did we discuss about LangGraph?",
            user_id="different_user",
            profile_id="different_profile"
        )
        
        print(f"✅ Answer: {result3['answer'][:100]}...")
        memory_context3 = result3.get('memory_context', {})
        if memory_context3.get('similar_messages'):
            print(f"⚠️  Unexpectedly found {len(memory_context3['similar_messages'])} messages in memory")
        else:
            print("✅ Correctly found no memory context for different user")
        
        # Test 4: Same user again (should see memory)
        print("\n📝 Test 4: Original user should still see memory")
        result4 = await agent.research(
            query="Can you remind me what we discussed about graph frameworks?",
            user_id=user_id,
            profile_id=profile_id
        )
        
        print(f"✅ Answer: {result4['answer'][:100]}...")
        memory_context4 = result4.get('memory_context', {})
        if memory_context4.get('similar_messages'):
            print(f"🧠 Found {len(memory_context4['similar_messages'])} similar messages from memory")
            for msg in memory_context4['similar_messages'][:2]:
                print(f"   • {msg['content'][:50]}... (similarity: {msg['similarity']:.2f})")
        else:
            print("⚠️  No memory context found")

    print("\n🎉 Memory functionality tests completed!")

if __name__ == "__main__":
    try:
        asyncio.run(test_memory_functionality())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()