from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from app.types.document_types import DocumentSource
from pydantic import BaseModel

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

class FetchMetadata(BaseModel):
    metadata: Optional[Any] = None


@dataclass
class PaginatedDocuments:
    documents: list[DocumentData]
    fetch_metadata: FetchMetadata
    has_more: bool = False