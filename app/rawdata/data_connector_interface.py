"""
Data Connector Interface for fetching documents from various sources.
Provides a standardized interface for different data sources to return Document objects.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.types.document_types import DocumentSource, Credentials
from app.dto.document_dto import DocumentData, PaginatedDocuments, FetchMetadata

class DataConnector(ABC):
    """
    Abstract base class for data connectors.
    
    All data connectors should implement this interface to provide
    a standardized way to fetch documents from different sources.
    """
    
    def __init__(self, source: DocumentSource, **kwargs):
        """
        Initialize the data connector.
        
        Args:
            source: The document source type
            **kwargs: Additional configuration parameters
        """
        self.source = source
        self.config = kwargs
    
    @abstractmethod
    async def get_documents(self, credentials: Credentials, metadata: FetchMetadata, **kwargs) -> PaginatedDocuments:
        """
        Fetch documents from the data source.
        
        Args:
            credentials: Credentials object containing authentication information
            **kwargs: Source-specific parameters (e.g., limit, offset, filters)
            
        Returns:
            PaginatedDocuments object containing documents and pagination info
            
        Raises:
            ConnectionError: If unable to connect to the data source
            ValueError: If required parameters are missing or invalid
        """
        pass
    
    @abstractmethod
    async def test_connection(self, credentials: Credentials) -> bool:
        """
        Test the connection to the data source.
        
        Args:
            credentials: Credentials object containing authentication information
            
        Returns:
            True if connection is successful, False otherwise
        """
        pass
    
    def generate_auth_headers(self, credentials: Credentials) -> Dict[str, str]:
        """
        Generate authorization headers based on credentials.
        
        Args:
            credentials: Credentials object containing authentication information
            
        Returns:
            Dictionary containing authorization headers
        """
        # Default implementation - override in subclasses for specific auth methods
        return {
            "Authorization": f"Bearer {credentials.api_key}",
            "Content-Type": "application/json"
        }
    
    def create_document_data(
        self,
        title: str,
        content: str,
        original_id: Optional[str] = None,
        content_url: Optional[str] = None,
        language: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ) -> DocumentData:
        """
        Create a DocumentData object with the connector's source.
        
        Args:
            title: Document title
            content: Document content
            original_id: Original ID from source system
            content_url: URL to original content
            language: Document language
            metadata: Additional metadata
            created_at: Creation timestamp
            updated_at: Last update timestamp
            
        Returns:
            DocumentData object
        """
        return DocumentData(
            title=title,
            content=content,
            source=self.source,
            original_id=original_id,
            content_url=content_url,
            language=language,
            metadata=metadata or {},
            created_at=created_at,
            updated_at=updated_at
        )
    
    def validate_document_data(self, doc_data: DocumentData) -> bool:
        """
        Validate document data before processing.
        
        Args:
            doc_data: DocumentData object to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not doc_data.title or not doc_data.title.strip():
            return False
        
        if not doc_data.content or not doc_data.content.strip():
            return False
        
        if not doc_data.source:
            return False
        
        return True
    
    async def get_documents_with_validation(self, credentials: Credentials, **kwargs) -> PaginatedDocuments:
        """
        Fetch documents and validate them before returning.
        
        Args:
            credentials: Credentials object containing authentication information
            **kwargs: Source-specific parameters
            
        Returns:
            PaginatedDocuments object with validated DocumentData and pagination info
        """
        paginated = await self.get_documents(credentials, **kwargs)
        validated_docs = [doc for doc in paginated.documents if self.validate_document_data(doc)]
        return type(paginated)(
            documents=validated_docs,
            fetch_metadata=getattr(paginated, 'fetch_metadata', None),
            has_more=paginated.has_more
        )


class MockDataConnector(DataConnector):
    """
    Mock data connector for testing purposes.
    Returns sample documents for development and testing.
    """
    
    def __init__(self, source: DocumentSource = DocumentSource.CUSTOM):
        super().__init__(source)
    
    async def test_connection(self, credentials: Credentials) -> bool:
        """Mock connection test always returns True."""
        return True
    
    async def get_documents(self, credentials: Credentials, **kwargs) -> PaginatedDocuments:
        """
        Return mock documents for testing.
        
        Args:
            credentials: Credentials object containing authentication information
            **kwargs: Ignored for mock connector
            
        Returns:
            PaginatedDocuments object containing mock documents and pagination info
        """
        mock_docs = [
            self.create_document_data(
                title="Sample Document 1",
                content="This is a sample document content for testing purposes. It contains various information that would typically be found in a real document.",
                original_id="mock_001",
                content_url="https://example.com/doc1",
                language="en",
                metadata={"author": "Test User", "category": "sample"}
            ),
            self.create_document_data(
                title="Sample Document 2", 
                content="Another sample document with different content. This helps test the system with multiple documents.",
                original_id="mock_002",
                content_url="https://example.com/doc2",
                language="en",
                metadata={"author": "Test User", "category": "sample"}
            ),
            self.create_document_data(
                title="Technical Guide",
                content="A technical guide document containing detailed instructions and procedures for various technical tasks.",
                original_id="mock_003",
                content_url="https://example.com/tech-guide",
                language="en",
                metadata={"author": "Tech Team", "category": "guide", "difficulty": "intermediate"}
            )
        ]
        
        return PaginatedDocuments(
            documents=mock_docs,
            fetch_metadata=None,
            has_more=False
        )


# Factory function for creating connectors
def create_connector(source: DocumentSource, **config) -> DataConnector:
    """
    Factory function to create appropriate data connector based on source.
    
    Args:
        source: Document source type
        **config: Configuration parameters for the connector
        
    Returns:
        Configured DataConnector instance
        
    Raises:
        ValueError: If source is not supported
    """
    if source == DocumentSource.INTERCOM_ARTICLE:
        from .intercom.intercom_connector import IntercomConnector
        return IntercomConnector(**config)
    elif source == DocumentSource.JIRA_TASK:
        from .jira.jira_connector import JiraConnector
        return JiraConnector(**config)
    # elif source == DocumentSource.CONFLUENCE_PAGE:
    #     from .confluence_connector import ConfluenceConnector
    #     return ConfluenceConnector(**config)
    elif source == DocumentSource.CUSTOM:
        return MockDataConnector(DocumentSource.CUSTOM)
    else:
        raise ValueError(f"Unsupported document source: {source}")
