"""
Utility functions and helper classes for data connectors.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from app.types.document_types import DocumentSource
from app.dto.document_dto import DocumentData, PaginatedDocuments
from .data_connector_interface import create_connector, DataConnector


class DocumentFetcher:
    """
    High-level utility class for fetching documents from multiple sources.
    """
    
    def __init__(self):
        self.connectors: Dict[DocumentSource, DataConnector] = {}
    
    def register_connector(self, source: DocumentSource, **config) -> None:
        """
        Register a data connector for a specific source.
        
        Args:
            source: Document source type
            **config: Configuration parameters for the connector
        """
        try:
            connector = create_connector(source, **config)
            self.connectors[source] = connector
        except Exception as e:
            print(f"Failed to register connector for {source}: {e}")
    
    async def test_connections(self) -> Dict[DocumentSource, bool]:
        """
        Test connections for all registered connectors.
        
        Returns:
            Dictionary mapping source to connection test result
        """
        results = {}
        for source, connector in self.connectors.items():
            try:
                results[source] = await connector.test_connection()
            except Exception as e:
                print(f"Connection test failed for {source}: {e}")
                results[source] = False
        
        return results
    
    async def fetch_from_source(self, source: DocumentSource, **kwargs) -> Optional[PaginatedDocuments]:
        """
        Fetch documents from a specific source.
        
        Args:
            source: Document source to fetch from
            **kwargs: Source-specific parameters
            
        Returns:
            PaginatedDocuments or None if source not registered/failed
        """
        if source not in self.connectors:
            print(f"No connector registered for {source}")
            return None
        
        try:
            result = await self.connectors[source].get_documents(**kwargs)
            return result
        except Exception as e:
            print(f"Failed to fetch documents from {source}: {e}")
            return None
    
    async def fetch_from_all_sources(self, **kwargs) -> Dict[DocumentSource, PaginatedDocuments]:
        """
        Fetch documents from all registered sources.
        
        Args:
            **kwargs: Parameters passed to all connectors
            
        Returns:
            Dictionary mapping source to fetched documents
        """
        results = {}
        
        # Create tasks for concurrent fetching
        tasks = []
        sources = []
        
        for source in self.connectors.keys():
            tasks.append(self.fetch_from_source(source, **kwargs))
            sources.append(source)
        
        # Execute all tasks concurrently
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for source, result in zip(sources, fetch_results):
            if isinstance(result, Exception):
                print(f"Error fetching from {source}: {result}")
                results[source] = PaginatedDocuments(documents=[], has_more=False)
            elif result is not None:
                results[source] = result
            else:
                results[source] = PaginatedDocuments(documents=[], has_more=False)
        
        return results


async def quick_fetch(source: DocumentSource, limit: int = 10, **config) -> List[DocumentData]:
    """
    Quick utility function to fetch documents from a single source.
    
    Args:
        source: Document source type
        limit: Maximum number of documents to fetch
        **config: Configuration parameters for the connector
        
    Returns:
        List of DocumentData objects
    """
    try:
        connector = create_connector(source, **config)
        
        # Test connection first
        if not await connector.test_connection():
            print(f"Connection test failed for {source}")
            return []
        
        # Fetch documents
        result = await connector.get_documents(limit=limit)
        return result.documents
        
    except Exception as e:
        print(f"Quick fetch failed for {source}: {e}")
        return []
