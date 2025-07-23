"""
Chat session management with memory capabilities.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .history import (
    create_chat_session, get_chat_session, add_chat_message,
    get_session_messages, get_recent_context, search_similar_messages,
    update_session_activity
)
from ..database.database import initialize_database
from .history import initialize_chat_history

logger = logging.getLogger(__name__)


class ChatSession:
    """Manages a single chat session with memory capabilities."""
    
    def __init__(
        self,
        user_id: str,
        profile_id: str,
        session_id: Optional[str] = None,
        session_name: Optional[str] = None,
        max_context_messages: int = 20,
        similarity_threshold: float = 0.8
    ):
        """
        Initialize chat session.
        
        Args:
            user_id: User identifier
            profile_id: Profile identifier within user
            session_id: Existing session ID or None for new session
            session_name: Name for new session
            max_context_messages: Maximum messages to keep in context
            similarity_threshold: Threshold for similarity search
        """
        self.user_id = user_id
        self.profile_id = profile_id
        self.session_id = session_id
        self.session_name = session_name
        self.max_context_messages = max_context_messages
        self.similarity_threshold = similarity_threshold
        
        # In-memory cache for current session
        self._current_messages: List[Dict[str, Any]] = []
        self._session_data: Optional[Dict[str, Any]] = None
        
    async def initialize(self):
        """Initialize the session - create new or load existing."""
        if self.session_id:
            # Load existing session
            self._session_data = await get_chat_session(self.session_id)
            if not self._session_data:
                raise ValueError(f"Session {self.session_id} not found")
            
            # Load recent messages
            self._current_messages = await get_session_messages(
                self.session_id,
                limit=self.max_context_messages,
                reverse_order=True
            )
            # Reverse to get chronological order
            self._current_messages.reverse()
            
        else:
            # Create new session
            self.session_id = await create_chat_session(
                user_id=self.user_id,
                profile_id=self.profile_id,
                session_name=self.session_name,
                metadata={"created_by": "research_agent"}
            )
            
            self._session_data = await get_chat_session(self.session_id)
            self._current_messages = []
        
        logger.info(f"Initialized chat session {self.session_id} for user {self.user_id}/{self.profile_id}")
    
    async def add_message(
        self,
        role: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        token_count: Optional[int] = None
    ) -> str:
        """
        Add a message to the session.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            embedding: Message embedding vector
            metadata: Additional metadata (sources, references, etc.)
            token_count: Token count for the message
        
        Returns:
            Message ID
        """
        message_id = await add_chat_message(
            session_id=self.session_id,
            user_id=self.user_id,
            profile_id=self.profile_id,
            role=role,
            content=content,
            embedding=embedding,
            metadata=metadata,
            token_count=token_count
        )
        
        # Add to in-memory cache
        message_data = {
            "id": message_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "profile_id": self.profile_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "token_count": token_count,
            "message_order": len(self._current_messages),
            "created_at": datetime.utcnow().isoformat()
        }
        
        self._current_messages.append(message_data)
        
        # Keep only recent messages in memory
        if len(self._current_messages) > self.max_context_messages:
            self._current_messages = self._current_messages[-self.max_context_messages:]
        
        return message_id
    
    def get_current_messages(self) -> List[Dict[str, Any]]:
        """Get current session messages from memory cache."""
        return self._current_messages.copy()
    
    def get_conversation_context(self, max_messages: Optional[int] = None) -> str:
        """
        Get conversation context as a formatted string.
        
        Args:
            max_messages: Maximum messages to include
        
        Returns:
            Formatted conversation context
        """
        messages = self._current_messages
        if max_messages:
            messages = messages[-max_messages:]
        
        context_parts = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    async def get_memory_context(
        self,
        query_embedding: Optional[List[float]] = None,
        max_similar: int = 3,
        max_recent: int = 5
    ) -> Dict[str, Any]:
        """
        Get memory context from chat history.
        
        Args:
            query_embedding: Current query embedding for similarity search
            max_similar: Maximum similar messages to retrieve
            max_recent: Maximum recent messages to retrieve
        
        Returns:
            Memory context with similar and recent messages
        """
        memory_context = {
            "current_session": self.get_current_messages(),
            "similar_messages": [],
            "recent_context": []
        }
        
        try:
            # Get similar messages from history (including current session for memory continuity)
            if query_embedding:
                similar_messages = await search_similar_messages(
                    embedding=query_embedding,
                    user_id=self.user_id,
                    profile_id=self.profile_id,
                    limit=max_similar,
                    threshold=self.similarity_threshold,
                    exclude_session=None  # Don't exclude current session for better memory continuity
                )
                memory_context["similar_messages"] = similar_messages
            
            # Get recent context from all sessions (including current)
            recent_context = await get_recent_context(
                user_id=self.user_id,
                profile_id=self.profile_id,
                session_id=None,  # All sessions
                max_messages=max_recent,
                max_age_hours=24
            )
            
            # Include current session messages but exclude the very latest message
            # (which is likely the current query being processed)
            current_messages = self.get_current_messages()
            if current_messages:
                # Keep all but the last message from current session
                recent_from_current = current_messages[:-1] if len(current_messages) > 1 else []
                
                # Convert to the expected format
                for msg in recent_from_current:
                    recent_context.append({
                        "id": msg["id"],
                        "session_id": msg["session_id"],
                        "role": msg["role"],
                        "content": msg["content"],
                        "metadata": msg.get("metadata", {}),
                        "created_at": msg["created_at"]
                    })
            
            # Sort by creation time (most recent first) and limit
            recent_context = sorted(
                recent_context, 
                key=lambda x: x["created_at"], 
                reverse=True
            )[:max_recent]
            
            memory_context["recent_context"] = recent_context
            
        except Exception as e:
            logger.error(f"Error retrieving memory context: {e}")
        
        return memory_context
    
    def format_memory_for_prompt(self, memory_context: Dict[str, Any]) -> str:
        """
        Format memory context for inclusion in LLM prompt.
        
        Args:
            memory_context: Memory context from get_memory_context()
        
        Returns:
            Formatted memory string
        """
        context_parts = []
        
        # Current session context
        if memory_context["current_session"]:
            current_context = []
            for msg in memory_context["current_session"][-10:]:  # Last 10 messages
                role = msg["role"].upper()
                content = msg["content"][:200]  # Truncate long messages
                current_context.append(f"{role}: {content}")
            
            if current_context:
                context_parts.append(f"Current Conversation:\n{chr(10).join(current_context)}")
        
        # Similar past messages
        if memory_context["similar_messages"]:
            similar_context = []
            for msg in memory_context["similar_messages"]:
                content = msg["content"][:150]
                similarity = msg["similarity"]
                similar_context.append(f"- {content} (similarity: {similarity:.2f})")
            
            if similar_context:
                context_parts.append(f"Related Past Messages:\n{chr(10).join(similar_context)}")
        
        # Recent context from other sessions
        if memory_context["recent_context"]:
            recent_context = []
            for msg in memory_context["recent_context"][:3]:  # Top 3
                role = msg["role"].upper()
                content = msg["content"][:100]
                recent_context.append(f"- {role}: {content}")
            
            if recent_context:
                context_parts.append(f"Recent Activity:\n{chr(10).join(recent_context)}")
        
        return "\n\n".join(context_parts)
    
    async def close_session(self):
        """Close the session and perform cleanup."""
        # Update session activity
        if self.session_id:
            await update_session_activity(self.session_id)
        
        # Clear in-memory cache
        self._current_messages.clear()
        self._session_data = None
        
        logger.info(f"Closed chat session {self.session_id}")
    
    @property
    def session_info(self) -> Dict[str, Any]:
        """Get session information."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "profile_id": self.profile_id,
            "session_name": self.session_name,
            "message_count": len(self._current_messages),
            "session_data": self._session_data
        }


class ChatMemoryManager:
    """Manages multiple chat sessions and provides memory services."""
    
    def __init__(self):
        """Initialize chat memory manager."""
        self._active_sessions: Dict[str, ChatSession] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the memory manager."""
        if not self._initialized:
            # Initialize database connections
            await initialize_database()
            await initialize_chat_history()
            self._initialized = True
            logger.info("Chat memory manager initialized")
    
    async def get_or_create_session(
        self,
        user_id: str,
        profile_id: str,
        session_id: Optional[str] = None,
        session_name: Optional[str] = None,
        **kwargs
    ) -> ChatSession:
        """
        Get existing session or create new one.
        
        Args:
            user_id: User identifier
            profile_id: Profile identifier
            session_id: Optional existing session ID
            session_name: Optional session name for new sessions
            **kwargs: Additional session parameters
        
        Returns:
            ChatSession instance
        """
        if not self._initialized:
            await self.initialize()
        
        # Use existing session if available
        session_key = session_id or f"{user_id}_{profile_id}_new"
        
        if session_key in self._active_sessions:
            return self._active_sessions[session_key]
        
        # Create new session
        session = ChatSession(
            user_id=user_id,
            profile_id=profile_id,
            session_id=session_id,
            session_name=session_name,
            **kwargs
        )
        
        await session.initialize()
        self._active_sessions[session_key] = session
        
        return session
    
    async def close_session(self, session_id: str):
        """Close and remove a session from active sessions."""
        session_key = None
        for key, session in self._active_sessions.items():
            if session.session_id == session_id:
                session_key = key
                break
        
        if session_key:
            session = self._active_sessions.pop(session_key)
            await session.close_session()
    
    async def cleanup_inactive_sessions(self, max_inactive_minutes: int = 60):
        """Clean up inactive sessions from memory."""
        # This is a simple cleanup - in production you might want more sophisticated logic
        to_remove = []
        
        for key, session in self._active_sessions.items():
            # Simple heuristic: if no recent activity, mark for removal
            # You could extend this with actual timestamps
            if len(session._current_messages) == 0:
                to_remove.append(key)
        
        for key in to_remove:
            session = self._active_sessions.pop(key)
            await session.close_session()
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} inactive sessions")
    
    def get_active_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._active_sessions)
    
    async def get_user_session_stats(self, user_id: str, profile_id: str) -> Dict[str, Any]:
        """Get statistics for a user's sessions."""
        from .history import get_chat_statistics
        return await get_chat_statistics(user_id, profile_id)


# Global memory manager instance
memory_manager = ChatMemoryManager()