"""
Intercom data connector implementation.
Fetches articles and conversations from Intercom API.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
from ..data_connector_interface import DataConnector, DocumentData, DocumentSource
from app.dto.document_dto import PaginatedDocuments, FetchMetadata
from app.types.document_types import Credentials

class IntercomConnector(DataConnector):
    """
    Intercom data connector for fetching articles and conversations.
    """
    
    def __init__(self, base_url: str = "https://api.intercom.io"):
        """
        Initialize Intercom connector.
        
        Args:
            base_url: Base URL for Intercom API
        """
        super().__init__(DocumentSource.INTERCOM_ARTICLE)
        self.base_url = base_url.rstrip('/')
    
    def generate_auth_headers(self, credentials: Credentials) -> Dict[str, str]:
        """
        Generate authorization headers for Intercom API.
        
        Args:
            credentials: Credentials object containing the API key
            
        Returns:
            Dictionary containing authorization headers
        """
        return {
            "Authorization": f"Bearer {credentials.api_key}",
            "Intercom-Version": "2.13",
            "Content-Type": "application/json"
        }
    
    async def test_connection(self, credentials: Credentials) -> bool:
        """Test connection to Intercom API."""
        try:
            headers = self.generate_auth_headers(credentials)
            response = requests.get(f"{self.base_url}/articles", headers=headers, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def _extract_clean_text(self, html_content: str) -> str:
        """
        Extract clean text from HTML content.
        
        Args:
            html_content: Raw HTML string
            
        Returns:
            Clean text content
        """
        if not html_content:
            return ""
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    async def get_documents(self, credentials: Credentials, metadata: FetchMetadata,  **kwargs) -> PaginatedDocuments:
        """
        Fetch documents from Intercom.
        
        Args:
            credentials: Credentials object containing the API key
            **kwargs: Optional parameters
                - limit: Maximum number of articles to fetch
                - include_conversations: Whether to include conversations
                - fetch_metadata: Pagination info for fetching next page
                
        Returns:
            PaginatedDocuments object containing documents and pagination info
        """
        documents = []
        limit = metadata.metadata.get("limit", 100) if metadata.metadata else 100
        fetch_metadata = metadata.metadata.get("next_page_info") if metadata.metadata else None
        # Fetch articles and pagination info
        articles, page_info = await self._fetch_articles(credentials, limit, fetch_metadata)
        for article in articles:
            # Extract clean text from HTML body
            clean_content = self._extract_clean_text(article.get('body', ''))
            
            doc_data = self.create_document_data(
                title=article.get('title', ''),
                content=clean_content,  # Use clean text instead of raw HTML
                original_id=str(article.get('id', '')),
                content_url=article.get('url'),
                language=article.get('default_locale', 'en'),
                metadata={
                    'author_id': article.get('author_id'),
                    'state': article.get('state'),
                    'tags': article.get('tags', {}),
                    'type': 'article',
                    'original_html': article.get('body', '')  # Store original HTML in metadata if needed
                },
                created_at=datetime.fromtimestamp(article.get('created_at', 0)),
                updated_at=datetime.fromtimestamp(article.get('updated_at', 0))
            )
            documents.append(doc_data)
        
        # Fetch conversations if requested
        if kwargs.get('include_conversations', False):
            conversations = await self._fetch_conversations(credentials, kwargs.get('limit', 50))
            for conv in conversations:
                doc_data = self.create_document_data(
                    title=conv.get('title', f"Conversation {conv.get('id', '')}"),
                    content=conv.get('body', ''),
                    original_id=str(conv.get('id', '')),
                    content_url=None,  # Conversations don't have direct URLs
                    language='en',
                    metadata={
                        'author': conv.get('author', {}),
                        'assignee': conv.get('assignee'),
                        'priority': conv.get('priority'),
                        'tags': conv.get('tags', {}),
                        'type': 'conversation'
                    },
                    created_at=datetime.fromtimestamp(conv.get('created_at', 0)),
                    updated_at=datetime.fromtimestamp(conv.get('updated_at', 0))
                )
                documents.append(doc_data)
        
        has_more = bool(page_info)
        return PaginatedDocuments(
            documents=documents,
            fetch_metadata=FetchMetadata(metadata={"next_page_info": page_info}),
            has_more=has_more
        )
    
    async def _fetch_articles(self, credentials: Credentials, limit: int = 100, next_page_info: Optional[Any] = None) -> tuple[list[Dict[str, Any]], Optional[Any]]:
        """Fetch articles from Intercom API with pagination info."""
        try:
            headers = self.generate_auth_headers(credentials)
            params = {'limit': limit}
            if next_page_info:
                # Intercom uses 'starting_after' for cursor-based pagination
                params['starting_after'] = next_page_info
            response = requests.get(
                f"{self.base_url}/articles",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            data = response.json()
            articles = data.get('data', [])
            # Intercom pagination info is in 'pages' object
            page_info = data.get('pages', {}).get('next')
            return articles, page_info
        except Exception as e:
            print(f"Error fetching Intercom articles: {e}")
            return [], None
    
    async def _fetch_conversations(self, credentials: Credentials, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch conversations from Intercom API."""
        try:
            headers = self.generate_auth_headers(credentials)
            response = requests.get(
                f"{self.base_url}/conversations",
                headers=headers,
                params={'limit': limit}
            )
            response.raise_for_status()
            data = response.json()
            return data.get('conversations', [])
        except Exception as e:
            print(f"Error fetching Intercom conversations: {e}")
            return [] 