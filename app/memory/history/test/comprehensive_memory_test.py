#!/usr/bin/env python3
"""
Comprehensive test to verify memory retrieval works across multiple interactions.
"""

import asyncio
import logging
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.agents.research_agent.research_agent import ResearchAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_comprehensive_memory():
    """Test comprehensive memory retrieval scenarios."""
    logger.info("🔬 Running comprehensive memory tests...")
    
    user_id = "test_comprehensive"
    profile_id = "test_profile_comp"
    
    try:
        # Create research agent
        agent = ResearchAgent(
            model="gpt-4",
            enable_memory=True,
            enable_web_search=False,
            enable_kg=False
        )
        
        # Test 1: Basic name introduction and recall
        logger.info("=" * 50)
        logger.info("TEST 1: Basic name introduction and recall")
        logger.info("=" * 50)
        
        result1 = await agent.research(
            query="Hello, my name is Jackie Ors and I work as a software engineer",
            user_id=user_id,
            profile_id=profile_id
        )
        logger.info(f"Response 1: {result1['answer'][:100]}...")
        
        result2 = await agent.research(
            query="What is my name?",
            user_id=user_id,
            profile_id=profile_id
        )
        logger.info(f"Response 2: {result2['answer']}")
        
        test1_pass = "Jackie" in result2['answer'] and len(result2.get('memory_context', {}).get('similar_messages', [])) > 0
        logger.info(f"Test 1 Result: {'✅ PASS' if test1_pass else '❌ FAIL'}")
        
        # Test 2: Job recall
        logger.info("=" * 50)
        logger.info("TEST 2: Job information recall")
        logger.info("=" * 50)
        
        result3 = await agent.research(
            query="What do I do for work?",
            user_id=user_id,
            profile_id=profile_id
        )
        logger.info(f"Response 3: {result3['answer']}")
        
        test2_pass = ("software engineer" in result3['answer'].lower() or "engineer" in result3['answer'].lower()) and len(result3.get('memory_context', {}).get('similar_messages', [])) > 0
        logger.info(f"Test 2 Result: {'✅ PASS' if test2_pass else '❌ FAIL'}")
        
        # Test 3: Multiple facts recall
        logger.info("=" * 50)
        logger.info("TEST 3: Multiple facts recall")
        logger.info("=" * 50)
        
        result4 = await agent.research(
            query="I also love hiking and have a dog named Max",
            user_id=user_id,
            profile_id=profile_id
        )
        logger.info(f"Response 4: {result4['answer'][:100]}...")
        
        result5 = await agent.research(
            query="Tell me about my hobbies and pets",
            user_id=user_id,
            profile_id=profile_id
        )
        logger.info(f"Response 5: {result5['answer']}")
        
        test3_pass = ("hiking" in result5['answer'].lower() or "max" in result5['answer']) and len(result5.get('memory_context', {}).get('similar_messages', [])) > 0
        logger.info(f"Test 3 Result: {'✅ PASS' if test3_pass else '❌ FAIL'}")
        
        # Test 4: Memory context details
        logger.info("=" * 50)
        logger.info("TEST 4: Memory context analysis")
        logger.info("=" * 50)
        
        memory_context = result5.get('memory_context', {})
        similar_msgs = memory_context.get('similar_messages', [])
        current_msgs = memory_context.get('current_session', [])
        
        logger.info(f"Similar messages found: {len(similar_msgs)}")
        logger.info(f"Current session messages: {len(current_msgs)}")
        
        for i, msg in enumerate(similar_msgs[:3]):  # Show top 3
            logger.info(f"  Similar {i+1}: '{msg['content'][:80]}...' (sim: {msg.get('similarity', 'N/A'):.4f})")
        
        test4_pass = len(similar_msgs) > 0 and len(current_msgs) > 0
        logger.info(f"Test 4 Result: {'✅ PASS' if test4_pass else '❌ FAIL'}")
        
        # Summary
        logger.info("=" * 50)
        logger.info("COMPREHENSIVE TEST SUMMARY")
        logger.info("=" * 50)
        
        all_tests_pass = test1_pass and test2_pass and test3_pass and test4_pass
        
        logger.info(f"Test 1 (Name recall): {'✅ PASS' if test1_pass else '❌ FAIL'}")
        logger.info(f"Test 2 (Job recall): {'✅ PASS' if test2_pass else '❌ FAIL'}")
        logger.info(f"Test 3 (Hobby/Pet recall): {'✅ PASS' if test3_pass else '❌ FAIL'}")
        logger.info(f"Test 4 (Memory context): {'✅ PASS' if test4_pass else '❌ FAIL'}")
        logger.info(f"Overall: {'🎉 ALL TESTS PASSED!' if all_tests_pass else '💥 SOME TESTS FAILED'}")
        
        return all_tests_pass
        
    except Exception as e:
        logger.error(f"❌ Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    success = await test_comprehensive_memory()
    return success


if __name__ == "__main__":
    asyncio.run(main())