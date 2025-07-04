from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from app.types.document_types import DocumentSource

@dataclass
class DocumentData:
    """Data transfer object for document information."""
    title: str
    content: str
    source: DocumentSource
    original_id: Optional[str] = None
    content_url: Optional[str] = None
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class PaginatedDocuments:
    documents: list[DocumentData]
    next_page_info: Optional[Any] = None
    has_more: bool = False