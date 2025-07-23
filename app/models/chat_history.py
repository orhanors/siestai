"""
Chat history SQLAlchemy models for Alembic migrations.
"""

from sqlalchemy import Column, UUID, Boolean, TIMESTAMP, Integer, Text, ForeignKey, text, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.memory.database.database import Base


class ChatSession(Base):
    """Chat session model with user and profile support."""
    
    __tablename__ = 'chat_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text('gen_random_uuid()'))
    user_id = Column(Text, nullable=False, comment='External user identifier')
    profile_id = Column(Text, nullable=False, comment='Profile identifier within user')
    session_name = Column(Text, nullable=True, comment='Optional session name')
    session_metadata = Column(JSONB, nullable=False, server_default=text("'{}'"), comment='Additional session metadata')
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('NOW()'))
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('NOW()'))
    is_active = Column(Boolean, nullable=False, server_default=text('true'))
    
    # Relationships
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """Individual chat message model with embeddings for semantic search."""
    
    __tablename__ = 'chat_messages'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text('gen_random_uuid()'))
    session_id = Column(UUID(as_uuid=True), ForeignKey('chat_sessions.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(Text, nullable=False, comment='Denormalized for faster queries')
    profile_id = Column(Text, nullable=False, comment='Denormalized for faster queries')
    role = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    message_metadata = Column(JSONB, nullable=False, server_default=text("'{}'"), comment='Sources, references, etc.')
    embedding = Column(Vector(1536), nullable=True, comment='Message embedding for semantic search')
    token_count = Column(Integer, nullable=True, comment='Token count for the message')
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('NOW()'))
    message_order = Column(Integer, nullable=False, comment='Order within session')
    
    # Constraints
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name='check_message_role'),
    )
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")


