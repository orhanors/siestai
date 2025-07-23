#!/usr/bin/env python3
"""
Test script to reproduce memory retention issues in long conversations.
"""

import asyncio
import logging
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from app.agents.research_agent.research_agent import ResearchAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_memory_retention():
    """Test memory retention over a longer conversation."""
    logger.info("🧪 Testing memory retention over long conversation...")
    
    user_id = "test_memory_retention_user"
    profile_id = "test_memory_retention_profile"
    
    try:
        # Create research agent
        agent = ResearchAgent(
            model="gpt-4",
            enable_memory=True,
            enable_web_search=False,
            enable_kg=False
        )
        
        logger.info("✅ Research agent created")
        
        # Simulate a long conversation
        async with agent:
            # Message 1: Introduce name
            logger.info("📝 Message 1: Introducing name...")
            result1 = await agent.research(
                query="Hello, my name is Alexandra and I'm a data scientist working on machine learning projects",
                user_id=user_id,
                profile_id=profile_id
            )
            logger.info(f"🤖 Response 1: {result1['answer'][:150]}...")
            
            # Message 2: Ask about name (should remember)
            logger.info("📝 Message 2: Testing immediate name recall...")
            result2 = await agent.research(
                query="What is my name?",
                user_id=user_id,
                profile_id=profile_id
            )
            logger.info(f"🤖 Response 2: {result2['answer']}")
            name_remembered_2 = "Alexandra" in result2['answer']
            logger.info(f"Name remembered in message 2: {name_remembered_2}")
            
            # Message 3-7: Add some other conversation
            topics = [
                "Tell me about neural networks",
                "What is supervised learning?", 
                "Explain gradient descent",
                "What are some common ML algorithms?",
                "How does cross-validation work?"
            ]
            
            for i, topic in enumerate(topics, 3):
                logger.info(f"📝 Message {i}: {topic}")
                result = await agent.research(
                    query=topic,
                    user_id=user_id,
                    profile_id=profile_id
                )
                logger.info(f"🤖 Response {i}: {result['answer'][:100]}...")
            
            # Message 8: Ask about name again (critical test)
            logger.info("📝 Message 8: Testing name recall after multiple messages...")
            result8 = await agent.research(
                query="What is my name and profession?",
                user_id=user_id,
                profile_id=profile_id
            )
            logger.info(f"🤖 Response 8: {result8['answer']}")
            name_remembered_8 = "Alexandra" in result8['answer']
            profession_remembered_8 = "data scientist" in result8['answer'].lower()
            
            logger.info(f"Name remembered in message 8: {name_remembered_8}")
            logger.info(f"Profession remembered in message 8: {profession_remembered_8}")
            
            # Message 9-12: More conversation
            more_topics = [
                "What is deep learning?",
                "Explain convolutional neural networks", 
                "What is natural language processing?",
                "Tell me about transformer models"
            ]
            
            for i, topic in enumerate(more_topics, 9):
                logger.info(f"📝 Message {i}: {topic}")
                result = await agent.research(
                    query=topic,
                    user_id=user_id,
                    profile_id=profile_id
                )
                logger.info(f"🤖 Response {i}: {result['answer'][:100]}...")
            
            # Message 13: Final memory test
            logger.info("📝 Message 13: Final memory test after long conversation...")
            result13 = await agent.research(
                query="Can you remind me what my name is and what I do for work?",
                user_id=user_id,
                profile_id=profile_id
            )
            logger.info(f"🤖 Response 13: {result13['answer']}")
            name_remembered_13 = "Alexandra" in result13['answer']
            profession_remembered_13 = "data scientist" in result13['answer'].lower()
            
            logger.info(f"Name remembered in message 13: {name_remembered_13}")
            logger.info(f"Profession remembered in message 13: {profession_remembered_13}")
            
            # Analyze memory context for the final query
            memory_context = result13.get('memory_context', {})
            similar_count = len(memory_context.get('similar_messages', []))
            recent_count = len(memory_context.get('recent_context', []))
            current_count = len(memory_context.get('current_session', []))
            
            logger.info(f"📊 Final memory context analysis:")
            logger.info(f"   Similar messages: {similar_count}")
            logger.info(f"   Recent context: {recent_count}")
            logger.info(f"   Current session: {current_count}")
            
            # Print similar messages to see what was retrieved
            if memory_context.get('similar_messages'):
                logger.info("🔍 Similar messages found:")
                for i, msg in enumerate(memory_context['similar_messages'][:3], 1):
                    content = msg.get('content', '')[:100]
                    similarity = msg.get('similarity', 0)
                    logger.info(f"   {i}. {content}... (sim: {similarity:.3f})")
            
            # Results summary
            logger.info("📋 Memory Retention Test Results:")
            logger.info(f"   Message 2 (immediate): Name={name_remembered_2}")
            logger.info(f"   Message 8 (after 5 msgs): Name={name_remembered_8}, Job={profession_remembered_8}")
            logger.info(f"   Message 13 (after 10 msgs): Name={name_remembered_13}, Job={profession_remembered_13}")
            
            # Test passes if name is remembered in later messages
            success = name_remembered_13 and profession_remembered_13
            
            if success:
                logger.info("✅ Memory retention test PASSED")
            else:
                logger.error("❌ Memory retention test FAILED - information was forgotten")
                
            return success
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    success = await test_memory_retention()
    if success:
        logger.info("🎉 Memory retention test PASSED!")
    else:
        logger.error("💥 Memory retention test FAILED!")


if __name__ == "__main__":
    asyncio.run(main())