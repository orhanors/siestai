"""
Utility functions and helper classes for data connectors.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from app.types.document_types import DocumentSource, Credentials
from app.dto.document_dto import DocumentData, PaginatedDocuments, FetchMetadata
from .data_connector_interface import create_connector, DataConnector
from app.utils.logger import logger, api_logger, ProgressLogger, StatusLogger
from app.utils.rate_limiter import global_rate_limiter, RateLimitConfig


class DocumentFetcher:
    """
    High-level utility class for fetching documents from multiple sources.
    """
    
    def __init__(self):
        self.connectors: Dict[DocumentSource, DataConnector] = {}
        logger.info("DocumentFetcher initialized")
    
    def register_connector(self, source: DocumentSource, **config) -> None:
        """
        Register a data connector for a specific source.
        
        Args:
            source: Document source type
            **config: Configuration parameters for the connector
        """
        try:
            logger.info(f"Registering connector for {source.value}")
            connector = create_connector(source, **config)
            self.connectors[source] = connector
            logger.success(f"Successfully registered connector for {source.value}")
        except Exception as e:
            logger.error(f"Failed to register connector for {source.value}: {e}")
            raise
    
    async def test_connections(self, credentials: Credentials) -> Dict[DocumentSource, bool]:
        """
        Test connections for all registered connectors.
        
        Args:
            credentials: Credentials object containing authentication information
            
        Returns:
            Dictionary mapping source to connection test result
        """
        logger.info(f"Testing connections for {len(self.connectors)} registered connectors")
        results = {}
        
        with ProgressLogger("Testing connections") as progress:
            task = progress.add_task("Connection tests", total=len(self.connectors))
            
            for source, connector in self.connectors.items():
                try:
                    api_logger.api(f"Testing connection for {source.value}")
                    connection_ok = await connector.test_connection(credentials)
                    results[source] = connection_ok
                    
                    if connection_ok:
                        logger.success(f"Connection test passed for {source.value}")
                    else:
                        logger.warning(f"Connection test failed for {source.value}")
                        
                except Exception as e:
                    logger.error(f"Connection test failed for {source.value}: {e}")
                    results[source] = False
                
                progress.update(task, advance=1)
        
        successful_connections = sum(1 for success in results.values() if success)
        logger.info(f"Connection tests completed: {successful_connections}/{len(results)} successful")
        
        return results
    
    async def fetch_from_source(self, source: DocumentSource, credentials: Credentials, metadata: FetchMetadata, rate_limit_config: Optional[RateLimitConfig] = None, **kwargs) -> Optional[PaginatedDocuments]:
        """
        Fetch documents from a specific source with rate limiting.
        
        Args:
            source: Document source to fetch from
            credentials: Credentials object containing authentication information
            metadata: Metadata for pagination and fetching
            rate_limit_config: Optional rate limit configuration
            **kwargs: Source-specific parameters
            
        Returns:
            PaginatedDocuments or None if source not registered/failed
        """
        if source not in self.connectors:
            logger.error(f"No connector registered for {source.value}")
            return None
        
        try:
            # Apply rate limiting before making request
            rate_limiter = global_rate_limiter.get_limiter(source.value, rate_limit_config)
            await rate_limiter.acquire()
            
            api_logger.api(f"Fetching documents from {source.value}")
            
            result = await self.connectors[source].get_documents(credentials, metadata, **kwargs)
            
            if result:
                doc_count = len(result.documents)
                has_more = result.has_more
                logger.success(f"Successfully fetched {doc_count} documents from {source.value} source (has_more: {has_more})")
                logger.debug(f"Metadata: {result.fetch_metadata}")
            else:
                logger.warning(f"No documents returned from {source.value}")
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch documents from {source.value}: {e}")
            return None
    
    async def fetch_from_all_sources(self, credentials: Credentials, **kwargs) -> Dict[DocumentSource, PaginatedDocuments]:
        """
        Fetch documents from all registered sources.
        
        Args:
            credentials: Credentials object containing authentication information
            **kwargs: Parameters passed to all connectors
            
        Returns:
            Dictionary mapping source to fetched documents
        """
        source_count = len(self.connectors)
        logger.info(f"Starting concurrent fetch from {source_count} sources")
        
        results = {}
        
        # Create tasks for concurrent fetching
        tasks = []
        sources = []
        
        for source in self.connectors.keys():
            tasks.append(self.fetch_from_source(source, credentials, **kwargs))
            sources.append(source)
        
        # Execute all tasks concurrently with progress tracking
        with ProgressLogger("Fetching from all sources") as progress:
            main_task = progress.add_task("Overall progress", total=source_count)
            
            # Execute all tasks concurrently
            fetch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            total_docs = 0
            successful_sources = 0
            
            for source, result in zip(sources, fetch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching from {source.value}: {result}")
                    results[source] = PaginatedDocuments(documents=[], has_more=False)
                elif result is not None:
                    results[source] = result
                    total_docs += len(result.documents)
                    successful_sources += 1
                else:
                    logger.warning(f"No result returned from {source.value}")
                    results[source] = PaginatedDocuments(documents=[], has_more=False)
                
                progress.update(main_task, advance=1)
        
        # Show summary
        logger.info(f"Batch fetch completed: {successful_sources}/{source_count} sources successful, {total_docs} total documents")
        
        # Show detailed results table
        self._show_fetch_summary(results)
        
        return results
    
    def _show_fetch_summary(self, results: Dict[DocumentSource, PaginatedDocuments]) -> None:
        """Show a summary table of fetch results."""
        summary_data = {}
        
        for source, result in results.items():
            doc_count = len(result.documents)
            has_more = result.has_more
            
            if doc_count > 0:
                status = "ok"
                details = f"{doc_count} documents" + (" (has more)" if has_more else " (complete)")
            else:
                status = "warning"
                details = "No documents fetched"
            
            summary_data[source.value] = {
                "status": status,
                "details": details
            }
        
        logger.info("Document Fetch Summary:")
        StatusLogger.show_status_table(summary_data)


async def quick_fetch(source: DocumentSource, credentials: Credentials, limit: int = 10, **config) -> List[DocumentData]:
    """
    Quick utility function to fetch documents from a single source.
    
    Args:
        source: Document source type
        credentials: Credentials object containing authentication information
        limit: Maximum number of documents to fetch
        **config: Configuration parameters for the connector
        
    Returns:
        List of DocumentData objects
    """
    logger.info(f"Quick fetch from {source.value} (limit: {limit})")
    
    try:
        connector = create_connector(source, **config)
        
        # Test connection first
        api_logger.api(f"Testing connection for {source.value}")
        if not await connector.test_connection(credentials):
            logger.error(f"Connection test failed for {source.value}")
            return []
        
        logger.success(f"Connection test passed for {source.value}")
        
        # Fetch documents
        api_logger.api(f"Fetching documents from {source.value}")
        result = await connector.get_documents(credentials, limit=limit)
        
        doc_count = len(result.documents)
        logger.success(f"Quick fetch completed: {doc_count} documents from {source.value}")
        
        return result.documents
        
    except Exception as e:
        logger.error(f"Quick fetch failed for {source.value}: {e}")
        return []
