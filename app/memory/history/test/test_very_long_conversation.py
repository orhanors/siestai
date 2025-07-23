#!/usr/bin/env python3
"""
Test memory retention in very long conversations (15+ messages).
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


async def test_very_long_conversation():
    """Test memory retention across a very long conversation."""
    logger.info("🧪 Testing memory retention in very long conversation...")
    
    user_id = "test_very_long_user"
    profile_id = "test_very_long_profile"
    
    try:
        # Create research agent
        agent = ResearchAgent(
            model="gpt-4",
            enable_memory=True,
            enable_web_search=False,
            enable_kg=False
        )
        
        async with agent:
            # Message 1: Introduce detailed personal info
            logger.info("📝 Message 1: Detailed introduction...")
            result1 = await agent.research(
                query="Hi! My name is Dr. Maria Rodriguez, I'm a biomedical engineer working at Stanford University on cancer research. I specialize in nanotechnology applications for drug delivery.",
                user_id=user_id,
                profile_id=profile_id
            )
            logger.info(f"🤖 Response 1: {result1['answer'][:100]}...")
            
            # Messages 2-12: Long technical conversation
            conversation_topics = [
                "What are the main types of machine learning algorithms?",
                "Explain how convolutional neural networks work",
                "What is the difference between precision and recall?",
                "How does backpropagation work in neural networks?",
                "What are some common preprocessing techniques for data?",
                "Explain the concept of overfitting and how to prevent it",
                "What is cross-validation and why is it important?",
                "Describe the differences between supervised and unsupervised learning",
                "What are ensemble methods in machine learning?",
                "How do you choose the right evaluation metric for a model?",
                "What is transfer learning and when would you use it?"
            ]
            
            for i, topic in enumerate(conversation_topics, 2):
                logger.info(f"📝 Message {i}: {topic[:50]}...")
                result = await agent.research(
                    query=topic,
                    user_id=user_id,
                    profile_id=profile_id
                )
                logger.info(f"🤖 Response {i}: {result['answer'][:60]}...")
            
            # Message 13: Test name recall after long conversation
            logger.info("📝 Message 13: Testing name recall after long conversation...")
            result13 = await agent.research(
                query="What is my full name?",
                user_id=user_id,
                profile_id=profile_id
            )
            logger.info(f"🤖 Response 13: {result13['answer']}")
            
            # Message 14: Test profession recall
            logger.info("📝 Message 14: Testing profession recall...")
            result14 = await agent.research(
                query="Where do I work and what is my specialty?",
                user_id=user_id,
                profile_id=profile_id
            )
            logger.info(f"🤖 Response 14: {result14['answer']}")
            
            # Message 15: Test complete personal info recall
            logger.info("📝 Message 15: Testing complete personal info recall...")
            result15 = await agent.research(
                query="Can you tell me about my background - my name, title, workplace, and research focus?",
                user_id=user_id,
                profile_id=profile_id
            )
            logger.info(f"🤖 Response 15: {result15['answer']}")
            
            # Analyze final results
            name_retained = "Maria Rodriguez" in result15['answer'] or ("Maria" in result15['answer'] and "Rodriguez" in result15['answer'])
            title_retained = "Dr." in result15['answer'] or "Doctor" in result15['answer']
            workplace_retained = "Stanford" in result15['answer']
            field_retained = any(word in result15['answer'].lower() for word in ['biomedical', 'cancer', 'nanotechnology', 'drug delivery'])
            
            logger.info(f"📊 Final retention analysis:")
            logger.info(f"   Name retained: {name_retained}")
            logger.info(f"   Title retained: {title_retained}")
            logger.info(f"   Workplace retained: {workplace_retained}")
            logger.info(f"   Research field retained: {field_retained}")
            
            # Memory context analysis
            memory_context = result15.get('memory_context', {})
            current_count = len(memory_context.get('current_session', []))
            recent_count = len(memory_context.get('recent_context', []))
            similar_count = len(memory_context.get('similar_messages', []))
            
            logger.info(f"📊 Memory context for final query (15 messages in):")
            logger.info(f"   Current session messages: {current_count}")
            logger.info(f"   Recent context messages: {recent_count}")
            logger.info(f"   Similar messages: {similar_count}")
            
            # Check if introduction is still accessible
            current_session = memory_context.get('current_session', [])
            intro_in_current = any('Maria Rodriguez' in msg.get('content', '') for msg in current_session)
            
            recent_context = memory_context.get('recent_context', [])  
            intro_in_recent = any('Maria Rodriguez' in msg.get('content', '') for msg in recent_context)
            
            logger.info(f"   Introduction found in current session: {intro_in_current}")
            logger.info(f"   Introduction found in recent context: {intro_in_recent}")
            
            # Test passes if core info is retained
            success = name_retained and workplace_retained and field_retained
            
            if success:
                logger.info("🎉 Very long conversation memory test PASSED!")
            else:
                logger.error("❌ Very long conversation memory test FAILED!")
                logger.error(f"Details - Name: {name_retained}, Workplace: {workplace_retained}, Field: {field_retained}")
                
            return success
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    success = await test_very_long_conversation()
    if success:
        logger.info("✅ Memory works great even in very long conversations!")
    else:
        logger.error("❌ Memory needs more work for very long conversations!")


if __name__ == "__main__":
    asyncio.run(main())