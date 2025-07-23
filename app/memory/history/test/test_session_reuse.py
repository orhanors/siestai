#!/usr/bin/env python3  
"""
Test script to verify session reuse behavior.
"""

import asyncio
import logging
import sys
import os
from uuid import uuid4

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.agents.research_agent.research_agent import ResearchAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_session_reuse():
    """Test session reuse with explicit session_id."""
    logger.info("🧪 Testing session reuse...")
    
    user_id = "test_session_reuse"
    profile_id = "test_profile_session"
    
    try:
        # Create research agent
        agent = ResearchAgent(
            model="gpt-4",
            enable_memory=True,
            enable_web_search=False,
            enable_kg=False
        )
        
        # Step 1: First interaction without session_id (creates new session)
        logger.info("Step 1: First interaction - no session_id specified")
        
        result1 = await agent.research(
            query="my name is Jackie Ors",
            user_id=user_id,
            profile_id=profile_id
            # No session_id specified - will create new session
        )
        
        logger.info(f"Response 1: {result1['answer'][:100]}...")
        
        # Extract session_id from the first interaction if available
        # The session_id should be accessible somehow... let me debug this
        
        # Step 2: Second interaction without session_id (creates ANOTHER new session)
        logger.info("Step 2: Second interaction - no session_id specified")
        
        result2 = await agent.research(
            query="what is my name",
            user_id=user_id,
            profile_id=profile_id
            # No session_id specified - will create ANOTHER new session
        )
        
        logger.info(f"Response 2: {result2['answer']}")
        logger.info(f"Memory context: {len(result2.get('memory_context', {}).get('similar_messages', []))} similar messages")
        
        # Step 3: Let's try creating a consistent session approach
        logger.info("Step 3: Testing with explicit session management")
        
        # Create a specific session_id to reuse (must be valid UUID)
        session_id = str(uuid4())
        
        logger.info("3a: First interaction with explicit session_id")
        result3 = await agent.research(
            query="my name is Jackie Ors and I'm a developer",
            user_id=user_id + "_v2", 
            profile_id=profile_id + "_v2",
            session_id=session_id
        )
        logger.info(f"Response 3: {result3['answer'][:100]}...")
        
        logger.info("3b: Second interaction with SAME session_id")
        result4 = await agent.research(
            query="what is my name and job",
            user_id=user_id + "_v2",
            profile_id=profile_id + "_v2", 
            session_id=session_id  # Same session_id!
        )
        logger.info(f"Response 4: {result4['answer']}")
        logger.info(f"Memory context: {len(result4.get('memory_context', {}).get('similar_messages', []))} similar messages")
        
        # Check if Jackie is mentioned in the response
        if "Jackie" in result4['answer']:
            logger.info("✅ SUCCESS: Agent remembered the name with consistent session!")
        else:
            logger.error("❌ FAILURE: Agent did not remember the name even with consistent session")
        
        return "Jackie" in result4['answer']
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    success = await test_session_reuse()
    if success:
        logger.info("🎉 Session reuse test PASSED!")
    else:
        logger.error("💥 Session reuse test FAILED!")


if __name__ == "__main__":
    asyncio.run(main())