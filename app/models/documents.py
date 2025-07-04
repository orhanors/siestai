from sqlalchemy import Column, String, Text, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import uuid
from app.database.database import Base

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_id = Column(Text, nullable=True) # e.g. intercom_article_id, jira_task_id
    content_url = Column(Text, nullable=True) # e.g. intercom_article_url, jira_task_url
    source = Column(Text, nullable=False)  # e.g., "intercom_article", "jira_task"
    title = Column(Text, nullable=True)
    content = Column(Text, nullable=False)
    language = Column(Text, nullable=True)
    doc_metadata = Column(JSONB, nullable=True)  # tags, timestamps, authors, etc.
    embedding = Column(Vector(1536), nullable=True)  # adjust dimension to your model
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<Document(id={self.id}, source='{self.source.value}', title='{self.title}')>"
    
    @property
    def source_value(self) -> str:
        """Get the string value of the source enum"""
        return self.source.value if self.source else None