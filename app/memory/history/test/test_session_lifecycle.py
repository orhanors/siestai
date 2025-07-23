#!/usr/bin/env python3
"""
Test script to verify complete session lifecycle including proper closure.
"""

import asyncio
import logging
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from app.agents.research_agent.research_agent import ResearchAgent
from app.memory.history.history import get_chat_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_session_lifecycle():
    """Test complete session lifecycle including proper closure."""
    logger.info("🧪 Testing complete session lifecycle...")
    
    user_id = "test_lifecycle_user"
    profile_id = "test_lifecycle_profile"
    
    try:
        # Create research agent with memory enabled
        agent = ResearchAgent(
            model="gpt-4",
            enable_memory=True,
            enable_web_search=False,
            enable_kg=False
        )
        
        logger.info("✅ Research agent created")
        
        # Simulate a conversation
        async with agent:
            # First interaction
            logger.info("📝 First interaction...")
            result1 = await agent.research(
                query="My name is Alice and I like AI research",
                user_id=user_id,
                profile_id=profile_id
            )
            
            # Get the session ID from the agent's internal tracking
            session_id = getattr(agent, '_last_session_id', None)
            if not session_id:
                # Find the session through memory manager if not tracked
                session_key = f"{user_id}_{profile_id}_new"
                if session_key in agent.memory_manager._active_sessions:
                    session_id = agent.memory_manager._active_sessions[session_key].session_id
            
            logger.info(f"Session ID: {session_id}")
            
            # Check that session is active
            session_data = await get_chat_session(session_id)
            if session_data and session_data['is_active']:
                logger.info("✅ Session is active after first interaction")
            else:
                logger.error("❌ Session is not active - this is wrong!")
                return False
            
            # Second interaction
            logger.info("📝 Second interaction...")
            result2 = await agent.research(
                query="What is my name?",
                user_id=user_id,
                profile_id=profile_id
            )
            
            # Check if Alice is mentioned in the response
            if "Alice" in result2['answer']:
                logger.info("✅ Agent remembered the name correctly")
            else:
                logger.warning("⚠️ Agent didn't remember the name (but this is a model issue, not session)")
            
            # Manually close the session (simulating user exit)
            logger.info("🔒 Manually closing session...")
            await agent.close_current_session(user_id, profile_id, session_id)
        
        # The async context manager should also close the session, but it's already closed
        
        # Check that session is now inactive
        session_data = await get_chat_session(session_id)
        if session_data and not session_data['is_active']:
            logger.info("✅ Session is now marked as inactive after closure")
            return True
        else:
            logger.error("❌ Session is still active after closure - BUG!")
            logger.error(f"Session data: {session_data}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_context_manager_closure():
    """Test that context manager properly closes sessions."""
    logger.info("🧪 Testing context manager session closure...")
    
    user_id = "test_context_user"
    profile_id = "test_context_profile"
    session_id = None
    
    try:
        # Use agent in context manager
        async with ResearchAgent(
            model="gpt-4",
            enable_memory=True,
            enable_web_search=False,
            enable_kg=False
        ) as agent:
            
            # Single interaction
            result = await agent.research(
                query="Hello, I'm testing context manager closure",
                user_id=user_id,
                profile_id=profile_id
            )
            
            # Get session ID
            session_key = f"{user_id}_{profile_id}_new"
            if session_key in agent.memory_manager._active_sessions:
                session_id = agent.memory_manager._active_sessions[session_key].session_id
                logger.info(f"Session created: {session_id}")
        
        # Context manager should have closed the session
        if session_id:
            session_data = await get_chat_session(session_id)
            if session_data and not session_data['is_active']:
                logger.info("✅ Context manager properly closed the session")
                return True
            else:
                logger.error("❌ Context manager did not close the session")
                return False
        else:
            logger.error("❌ No session was created")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all lifecycle tests."""
    logger.info("🚀 Starting session lifecycle tests...")
    
    # Test 1: Manual session closure
    success1 = await test_session_lifecycle()
    
    # Test 2: Context manager closure
    success2 = await test_context_manager_closure()
    
    if success1 and success2:
        logger.info("🎉 All session lifecycle tests PASSED!")
    else:
        logger.error("💥 Some session lifecycle tests FAILED!")
        logger.error(f"Manual closure: {'PASS' if success1 else 'FAIL'}")
        logger.error(f"Context manager: {'PASS' if success2 else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())