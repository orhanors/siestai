#!/usr/bin/env python3
"""
Test script to verify the new credentials-based approach works correctly.
"""

import asyncio
import os
from app.rawdata.document_fetcher import DocumentFetcher
from app.rawdata.data_connector_interface import create_connector
from app.types.document_types import DocumentSource, Credentials
from app.utils.logger import logger

async def test_credentials_integration():
    """Test the new credentials-based approach."""
    logger.info("🧪 Testing credentials-based integration...")
    
    # Create test credentials
    credentials = Credentials(api_key=os.getenv("INTERCOM_ACCESS_TOKEN"))
    
    # Test 1: Direct connector usage
    logger.info("1️⃣ Testing direct connector usage...")
    try:
        # Create Intercom connector
        connector = create_connector(DocumentSource.INTERCOM_ARTICLE)
        
        # Test connection
        connection_ok = await connector.test_connection(credentials)
        logger.info(f"Connection test result: {connection_ok}")
        
        # Test document fetching (should use mock data)
        result = await connector.get_documents(credentials, limit=5)
        logger.info(f"Fetched {len(result.documents)} documents")
        
        if result.documents:
            sample_doc = result.documents[0]
            logger.info(f"Sample document: {sample_doc.title}")
            logger.info(f"Content length: {len(sample_doc.content)} characters")
        
    except Exception as e:
        logger.error(f"❌ Direct connector test failed: {e}")
    
    # Test 2: DocumentFetcher usage
    logger.info("2️⃣ Testing DocumentFetcher usage...")
    try:
        fetcher = DocumentFetcher()
        fetcher.register_connector(DocumentSource.INTERCOM_ARTICLE)
        fetcher.register_connector(DocumentSource.CUSTOM)
        
        # Test connection
        connection_results = await fetcher.test_connections(credentials)
        logger.info(f"Connection results: {connection_results}")
        
        # Test fetching from specific source
        result = await fetcher.fetch_from_source(DocumentSource.INTERCOM_ARTICLE, credentials, limit=3)
        if result:
            logger.info(f"Fetched {len(result.documents)} documents from Intercom")
        
        # Test fetching from all sources
        all_results = await fetcher.fetch_from_all_sources(credentials, limit=2)
        logger.info(f"Fetched from {len(all_results)} sources")
        
    except Exception as e:
        logger.error(f"❌ DocumentFetcher test failed: {e}")
    
    # Test 3: Quick fetch utility
    logger.info("3️⃣ Testing quick fetch utility...")
    try:
        from app.rawdata.document_fetcher import quick_fetch
        
        documents = await quick_fetch(DocumentSource.INTERCOM_ARTICLE, credentials, limit=2)
        logger.info(f"Quick fetch returned {len(documents)} documents")
        
    except Exception as e:
        logger.error(f"❌ Quick fetch test failed: {e}")
    
    logger.info("✅ Credentials integration test completed!")

if __name__ == "__main__":
    asyncio.run(test_credentials_integration()) 