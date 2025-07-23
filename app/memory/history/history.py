"""
Chat history database operations with user and profile support.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from uuid import uuid4
import logging
from datetime import datetime, timedelta
import asyncpg
from asyncpg.pool import Pool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ChatHistoryClient:
    """Chat history database client with user/profile support."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize chat history database pool.
        
        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            # Fallback to individual environment variables
            db_user = os.getenv("DB_USER", "siestai_user")
            db_password = os.getenv("DB_PASSWORD", "siestai_password")
            db_host = os.getenv("DB_HOST", "localhost")
            db_port = os.getenv("DB_PORT", "5439")
            db_name = os.getenv("DB_NAME", "siestai_dev")
            
            self.database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        self.pool: Optional[Pool] = None
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Create connection pool with retry logic."""
        async with self._lock:
            if not self.pool:
                try:
                    self.pool = await asyncpg.create_pool(
                        self.database_url,
                        min_size=5,
                        max_size=20,
                        max_inactive_connection_lifetime=300,
                        command_timeout=60,
                        setup=self._setup_connection
                    )
                    logger.info("Chat history database connection pool initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize chat history database pool: {e}")
                    raise
    
    async def _setup_connection(self, connection):
        """Setup connection with custom types and functions."""
        # Enable pgvector extension if not already enabled
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Set timezone
        await connection.execute("SET timezone = 'UTC'")
    
    async def close(self):
        """Close connection pool."""
        async with self._lock:
            if self.pool:
                await self.pool.close()
                self.pool = None
                logger.info("Chat history database connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool with automatic initialization."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as connection:
            yield connection


# Global chat history client instance
chat_history_client = ChatHistoryClient()


async def initialize_chat_history():
    """Initialize chat history database connection pool."""
    await chat_history_client.initialize()


async def close_chat_history():
    """Close chat history database connection pool."""
    await chat_history_client.close()


# Session Management Functions
async def create_chat_session(
    user_id: str,
    profile_id: str,
    session_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a new chat session.
    
    Args:
        user_id: User identifier
        profile_id: Profile identifier within user
        session_name: Optional session name
        metadata: Additional session metadata
    
    Returns:
        Session ID
    """
    async with chat_history_client.acquire() as conn:
        result = await conn.fetchrow(
            """
            INSERT INTO chat_sessions (
                id, user_id, profile_id, session_name, session_metadata
            )
            VALUES ($1::uuid, $2, $3, $4, $5::jsonb)
            RETURNING id::text
            """,
            uuid4(),
            user_id,
            profile_id,
            session_name,
            json.dumps(metadata or {})
        )
        
        return result["id"]


async def get_chat_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get chat session by ID.
    
    Args:
        session_id: Session UUID
    
    Returns:
        Session data or None if not found
    """
    async with chat_history_client.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT 
                id::text,
                user_id,
                profile_id,
                session_name,
                session_metadata,
                created_at,
                updated_at,
                is_active
            FROM chat_sessions
            WHERE id = $1::uuid
            """,
            session_id
        )
        
        if result:
            return {
                "id": result["id"],
                "user_id": result["user_id"],
                "profile_id": result["profile_id"],
                "session_name": result["session_name"],
                "metadata": json.loads(result["session_metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat(),
                "is_active": result["is_active"]
            }
        
        return None


async def list_user_sessions(
    user_id: str,
    profile_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    active_only: bool = True
) -> List[Dict[str, Any]]:
    """
    List chat sessions for a user/profile.
    
    Args:
        user_id: User identifier
        profile_id: Optional profile filter
        limit: Maximum sessions to return
        offset: Number of sessions to skip
        active_only: Only return active sessions
    
    Returns:
        List of session data
    """
    async with chat_history_client.acquire() as conn:
        conditions = ["user_id = $1"]
        params = [user_id]
        param_count = 1
        
        if profile_id:
            param_count += 1
            conditions.append(f"profile_id = ${param_count}")
            params.append(profile_id)
        
        if active_only:
            param_count += 1
            conditions.append(f"is_active = ${param_count}")
            params.append(True)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT 
                id::text,
                user_id,
                profile_id,
                session_name,
                session_metadata,
                created_at,
                updated_at,
                is_active
            FROM chat_sessions
            WHERE {where_clause}
            ORDER BY updated_at DESC
            LIMIT ${param_count + 1} OFFSET ${param_count + 2}
        """
        
        params.extend([limit, offset])
        results = await conn.fetch(query, *params)
        
        return [
            {
                "id": row["id"],
                "user_id": row["user_id"],
                "profile_id": row["profile_id"],
                "session_name": row["session_name"],
                "metadata": json.loads(row["session_metadata"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "is_active": row["is_active"]
            }
            for row in results
        ]


async def update_session_activity(session_id: str) -> bool:
    """
    Update session's last activity timestamp.
    
    Args:
        session_id: Session UUID
    
    Returns:
        True if updated successfully
    """
    async with chat_history_client.acquire() as conn:
        result = await conn.execute(
            "UPDATE chat_sessions SET updated_at = NOW() WHERE id = $1::uuid",
            session_id
        )
        return result.split()[-1] != "0"


# Message Management Functions
async def add_chat_message(
    session_id: str,
    user_id: str,
    profile_id: str,
    role: str,
    content: str,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    token_count: Optional[int] = None
) -> str:
    """
    Add a message to a chat session.
    
    Args:
        session_id: Session UUID
        user_id: User identifier
        profile_id: Profile identifier
        role: Message role (user, assistant, system)
        content: Message content
        embedding: Message embedding vector
        metadata: Additional message metadata
        token_count: Token count for the message
    
    Returns:
        Message ID
    """
    async with chat_history_client.acquire() as conn:
        # Get next message order
        message_order = await conn.fetchval(
            "SELECT COALESCE(MAX(message_order), 0) + 1 FROM chat_messages WHERE session_id = $1::uuid",
            session_id
        )
        
        # Convert embedding to PostgreSQL vector format
        embedding_str = None
        if embedding:
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        result = await conn.fetchrow(
            """
            INSERT INTO chat_messages (
                id, session_id, user_id, profile_id, role, content,
                message_metadata, embedding, token_count, message_order
            )
            VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7::jsonb, $8::vector, $9, $10)
            RETURNING id::text
            """,
            uuid4(),
            session_id,
            user_id,
            profile_id,
            role,
            content,
            json.dumps(metadata or {}),
            embedding_str,
            token_count,
            message_order
        )
        
        # Update session activity
        await update_session_activity(session_id)
        
        return result["id"]


async def get_session_messages(
    session_id: str,
    limit: Optional[int] = None,
    offset: int = 0,
    reverse_order: bool = False
) -> List[Dict[str, Any]]:
    """
    Get messages from a chat session.
    
    Args:
        session_id: Session UUID
        limit: Maximum messages to return
        offset: Number of messages to skip
        reverse_order: Return in reverse chronological order
    
    Returns:
        List of message data
    """
    async with chat_history_client.acquire() as conn:
        order_clause = "message_order DESC" if reverse_order else "message_order ASC"
        limit_clause = f"LIMIT {limit}" if limit else ""
        offset_clause = f"OFFSET {offset}" if offset > 0 else ""
        
        query = f"""
            SELECT 
                id::text,
                session_id::text,
                user_id,
                profile_id,
                role,
                content,
                message_metadata,
                token_count,
                message_order,
                created_at
            FROM chat_messages
            WHERE session_id = $1::uuid
            ORDER BY {order_clause}
            {limit_clause} {offset_clause}
        """
        
        results = await conn.fetch(query, session_id)
        
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "user_id": row["user_id"],
                "profile_id": row["profile_id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": json.loads(row["message_metadata"]),
                "token_count": row["token_count"],
                "message_order": row["message_order"],
                "created_at": row["created_at"].isoformat()
            }
            for row in results
        ]


async def get_recent_context(
    user_id: str,
    profile_id: str,
    session_id: Optional[str] = None,
    max_messages: int = 10,
    max_age_hours: int = 24
) -> List[Dict[str, Any]]:
    """
    Get recent conversation context for memory.
    
    Args:
        user_id: User identifier
        profile_id: Profile identifier
        session_id: Optional specific session ID
        max_messages: Maximum messages to return
        max_age_hours: Maximum age in hours
    
    Returns:
        List of recent messages
    """
    async with chat_history_client.acquire() as conn:
        conditions = ["user_id = $1", "profile_id = $2"]
        params = [user_id, profile_id]
        param_count = 2
        
        if session_id:
            param_count += 1
            conditions.append(f"session_id = ${param_count}::uuid")
            params.append(session_id)
        
        # Add time filter
        param_count += 1
        conditions.append(f"created_at > NOW() - INTERVAL '{max_age_hours} hours'")
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT 
                id::text,
                session_id::text,
                role,
                content,
                message_metadata,
                created_at
            FROM chat_messages
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT {max_messages}
        """
        
        results = await conn.fetch(query, *params)
        
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": json.loads(row["message_metadata"]),
                "created_at": row["created_at"].isoformat()
            }
            for row in results
        ]


# Semantic Search Functions
async def search_similar_messages(
    embedding: List[float],
    user_id: str,
    profile_id: str,
    limit: int = 5,
    threshold: float = 0.8,
    exclude_session: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for similar messages using vector similarity.
    
    Args:
        embedding: Query embedding vector
        user_id: User identifier
        profile_id: Profile identifier
        limit: Maximum results to return
        threshold: Minimum similarity threshold
        exclude_session: Optional session ID to exclude
    
    Returns:
        List of similar messages with similarity scores
    """
    async with chat_history_client.acquire() as conn:
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        conditions = ["user_id = $2", "profile_id = $3", "embedding IS NOT NULL"]
        params = [embedding_str, user_id, profile_id, threshold, limit]
        
        if exclude_session:
            conditions.append("session_id != $6::uuid")
            params.append(exclude_session)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT 
                id::text,
                session_id::text,
                role,
                content,
                message_metadata,
                embedding <=> $1::vector AS similarity,
                created_at
            FROM chat_messages
            WHERE {where_clause}
            AND (embedding <=> $1::vector) < $4
            ORDER BY embedding <=> $1::vector
            LIMIT $5
        """
        
        results = await conn.fetch(query, *params)
        
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": json.loads(row["message_metadata"]),
                "similarity": float(row["similarity"]),
                "created_at": row["created_at"].isoformat()
            }
            for row in results
        ]




# Utility Functions
async def get_chat_statistics(
    user_id: str,
    profile_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get chat statistics for a user/profile.
    
    Args:
        user_id: User identifier
        profile_id: Optional profile filter
    
    Returns:
        Statistics dictionary
    """
    async with chat_history_client.acquire() as conn:
        conditions = ["user_id = $1"]
        params = [user_id]
        
        if profile_id:
            conditions.append("profile_id = $2")
            params.append(profile_id)
        
        where_clause = " AND ".join(conditions)
        
        # Session stats
        session_stats = await conn.fetchrow(f"""
            SELECT 
                COUNT(*) as total_sessions,
                COUNT(CASE WHEN is_active THEN 1 END) as active_sessions
            FROM chat_sessions
            WHERE {where_clause}
        """, *params)
        
        # Message stats
        message_stats = await conn.fetchrow(f"""
            SELECT 
                COUNT(*) as total_messages,
                COUNT(CASE WHEN role = 'user' THEN 1 END) as user_messages,
                COUNT(CASE WHEN role = 'assistant' THEN 1 END) as assistant_messages,
                SUM(token_count) as total_tokens
            FROM chat_messages
            WHERE {where_clause}
        """, *params)
        
        return {
            "total_sessions": session_stats["total_sessions"],
            "active_sessions": session_stats["active_sessions"],
            "total_messages": message_stats["total_messages"],
            "user_messages": message_stats["user_messages"],
            "assistant_messages": message_stats["assistant_messages"],
            "total_tokens": message_stats["total_tokens"] or 0
        }


async def cleanup_old_sessions(days_old: int = 30) -> int:
    """
    Cleanup old inactive sessions.
    
    Args:
        days_old: Number of days to keep sessions
    
    Returns:
        Number of sessions cleaned up
    """
    async with chat_history_client.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE chat_sessions 
            SET is_active = false 
            WHERE is_active = true 
            AND updated_at < NOW() - INTERVAL '%s days'
            """,
            days_old
        )
        
        return int(result.split()[-1])


async def test_chat_history_connection() -> bool:
    """Test chat history database connection."""
    try:
        async with chat_history_client.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Chat history database connection test failed: {e}")
        return False