"""Chat history management with user and profile support."""

from .history import (
    initialize_chat_history,
    close_chat_history,
    create_chat_session,
    create_chat_session_with_id,
    get_chat_session,
    add_chat_message,
    get_session_messages,
    get_recent_context,
    search_similar_messages,
    get_chat_statistics,
    update_session_activity,
    close_chat_session
)

from .session_manager import (
    ChatSession,
    ChatMemoryManager,
    memory_manager
)

__all__ = [
    "initialize_chat_history",
    "close_chat_history", 
    "create_chat_session",
    "create_chat_session_with_id",
    "get_chat_session",
    "add_chat_message",
    "get_session_messages",
    "get_recent_context",
    "search_similar_messages",
    "get_chat_statistics",
    "update_session_activity",
    "close_chat_session",
    "ChatSession",
    "ChatMemoryManager", 
    "memory_manager"
]