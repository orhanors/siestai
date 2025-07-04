"""
Intercom data connector implementation.
Fetches articles and conversations from Intercom API.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
from ..data_connector_interface import DataConnector, DocumentData, DocumentSource


class IntercomConnector(DataConnector):
    """
    Intercom data connector for fetching articles and conversations.
    """
    
    def __init__(self, access_token: Optional[str] = None, base_url: str = "https://api.intercom.io"):
        """
        Initialize Intercom connector.
        
        Args:
            access_token: Intercom access token
            base_url: Base URL for Intercom API
        """
        super().__init__(DocumentSource.INTERCOM_ARTICLE)
        self.access_token = access_token or os.getenv("INTERCOM_ACCESS_TOKEN")
        if not self.access_token:
            raise ValueError("Intercom access token is required")
        
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Intercom-Version": "2.13",
            "Content-Type": "application/json"
        }
    
    async def test_connection(self) -> bool:
        """Test connection to Intercom API."""
        try:
            response = requests.get(f"{self.base_url}/articles", headers=self.headers, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    async def get_documents(self, **kwargs) -> List[DocumentData]:
        """
        Fetch documents from Intercom.
        
        Args:
            **kwargs: Optional parameters
                - limit: Maximum number of articles to fetch
                - include_conversations: Whether to include conversations
                
        Returns:
            List of DocumentData objects
        """
        documents = []
        
        # Fetch articles
        articles = await self._fetch_articles(kwargs.get('limit', 100))
        for article in articles:
            doc_data = self.create_document_data(
                title=article.get('title', ''),
                content=article.get('body', ''),
                original_id=str(article.get('id', '')),
                content_url=article.get('url'),
                language=article.get('default_locale', 'en'),
                metadata={
                    'author_id': article.get('author_id'),
                    'state': article.get('state'),
                    'tags': article.get('tags', {}),
                    'type': 'article'
                },
                created_at=datetime.fromtimestamp(article.get('created_at', 0)),
                updated_at=datetime.fromtimestamp(article.get('updated_at', 0))
            )
            documents.append(doc_data)
        
        # Fetch conversations if requested
        if kwargs.get('include_conversations', False):
            conversations = await self._fetch_conversations(kwargs.get('limit', 50))
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
        
        return documents
    
    async def _fetch_articles(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch articles from Intercom API."""
        try:
            response = requests.get(
                f"{self.base_url}/articles",
                headers=self.headers,
                params={'limit': limit}
            )
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except Exception as e:
            print(f"Error fetching Intercom articles: {e}")
            return []
    
    async def _fetch_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch conversations from Intercom API."""
        try:
            response = requests.get(
                f"{self.base_url}/conversations",
                headers=self.headers,
                params={'limit': limit}
            )
            response.raise_for_status()
            data = response.json()
            return data.get('conversations', [])
        except Exception as e:
            print(f"Error fetching Intercom conversations: {e}")
            return [] 