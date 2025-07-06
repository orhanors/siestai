#!/usr/bin/env python3
"""
Test script for ingesting Intercom data into Neo4j knowledge graph.
This script demonstrates the complete flow from data fetching to knowledge graph ingestion.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add the parent directory to the path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from app.memory.knowledge_graph.knowledge_graph import KGClient
from app.rawdata.intercom.intercom_connector import IntercomConnector
from app.utils.logger import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
logger = setup_logging(level="INFO")

class KGIngestionTest:
    """Test class for knowledge graph ingestion."""
    
    def __init__(self):
        """Initialize test with KG client and Intercom connector."""
        self.kg_client = None
        self.intercom_connector = None
        
    async def setup(self):
        """Setup test environment."""
        logger.info("Setting up test environment...")
        
        # Initialize KG client with local Neo4j settings
        # Override URI to use localhost instead of 127.0.0.1 (connection test shows localhost works)
        self.kg_client = KGClient(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD")
        )
        
        try:
            # Initialize Intercom connector
            self.intercom_connector = IntercomConnector()
            logger.info("Intercom connector initialized successfully")
        except ValueError as e:
            logger.warning(f"Intercom connector initialization failed: {e}")
            logger.info("Continuing with mock data for testing...")
            self.intercom_connector = None
        
        # Initialize KG client
        await self.kg_client.initialize()
        logger.info("KG client initialized successfully")
    
    async def teardown(self):
        """Cleanup test environment."""
        logger.info("Tearing down test environment...")
        if self.kg_client:
            await self.kg_client.close()
        logger.info("Test environment cleaned up")
    
    async def test_connection(self):
        """Test connections to both services."""
        logger.info("Testing connections...")
        
        # Test KG connection
        try:
            stats = await self.kg_client.get_graph_statistics()
            logger.info(f"✅ KG connection successful. Stats: {stats}")
        except Exception as e:
            logger.error(f"❌ KG connection failed: {e}")
            return False
        
        # Test Intercom connection
        if self.intercom_connector:
            try:
                connection_ok = await self.intercom_connector.test_connection()
                if connection_ok:
                    logger.info("✅ Intercom connection successful")
                else:
                    logger.warning("⚠️ Intercom connection failed")
            except Exception as e:
                logger.error(f"❌ Intercom connection test failed: {e}")
        
        return True
    
    async def fetch_intercom_data(self, limit: int = 10):
        """Fetch sample data from Intercom."""
        logger.info(f"Fetching {limit} documents from Intercom...")
        
        if not self.intercom_connector:
            logger.info("Using mock data since Intercom connector not available")
            return self._create_mock_data()
        
        try:
            # Fetch articles only for testing
            paginated_docs = await self.intercom_connector.get_documents(
                limit=limit,
                include_conversations=False
            )
            
            documents = paginated_docs.documents
            logger.info(f"✅ Fetched {len(documents)} documents from Intercom")
            
            # Log sample document info
            if documents:
                sample_doc = documents[0]
                logger.info(f"Sample document: {sample_doc.title[:50]}...")
                logger.info(f"Content length: {len(sample_doc.content)} characters")
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch Intercom data: {e}")
            logger.info("Using mock data for testing...")
            return self._create_mock_data()
    
    def _create_mock_data(self):
        """Create mock data for testing when Intercom is not available."""
        from app.rawdata.data_connector_interface import DocumentData, DocumentSource
        
        mock_documents = [
            DocumentData(
                title="Getting Started with Customer Support",
                content="This guide explains how to provide excellent customer support. Key principles include active listening, empathy, and quick resolution of issues. Always follow up with customers to ensure satisfaction.",
                original_id="mock_1",
                source=DocumentSource.INTERCOM_ARTICLE,
                content_url="https://example.com/support-guide",
                language="en",
                metadata={
                    "type": "article",
                    "author_id": "mock_author_1",
                    "state": "published",
                    "tags": ["support", "getting-started"]
                },
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            DocumentData(
                title="Product Feature Overview",
                content="Our product offers advanced analytics, real-time monitoring, and automated reporting. Users can create custom dashboards and set up alerts for important metrics.",
                original_id="mock_2",
                source=DocumentSource.INTERCOM_ARTICLE,
                content_url="https://example.com/product-features",
                language="en",
                metadata={
                    "type": "article",
                    "author_id": "mock_author_2",
                    "state": "published",
                    "tags": ["product", "features"]
                },
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            DocumentData(
                title="Troubleshooting Common Issues",
                content="Common issues include connection problems, slow performance, and login failures. For connection issues, check your internet connection and firewall settings. For performance issues, try clearing your browser cache.",
                original_id="mock_3",
                source=DocumentSource.INTERCOM_ARTICLE,
                content_url="https://example.com/troubleshooting",
                language="en",
                metadata={
                    "type": "article",
                    "author_id": "mock_author_1",
                    "state": "published",
                    "tags": ["troubleshooting", "support"]
                },
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        
        logger.info(f"✅ Created {len(mock_documents)} mock documents")
        return mock_documents
    
    async def ingest_documents(self, documents):
        """Ingest documents into the knowledge graph."""
        logger.info(f"Ingesting {len(documents)} documents into knowledge graph...")
        
        ingested_episodes = []
        
        for i, doc in enumerate(documents):
            try:
                # Create episode content combining title and content
                episode_content = f"Title: {doc.title}\n\nContent: {doc.content}"
                
                # Create episode ID
                episode_id = f"intercom_article_{doc.original_id}_{i}"
                
                # Create source description
                source_desc = f"Intercom article from {doc.source.value}"
                if doc.content_url:
                    source_desc += f" (URL: {doc.content_url})"
                
                # Add episode to knowledge graph
                await self.kg_client.add_episode(
                    episode_id=episode_id,
                    content=episode_content,
                    source=source_desc,
                    timestamp=doc.created_at,
                    metadata=doc.metadata
                )
                
                ingested_episodes.append(episode_id)
                logger.info(f"✅ Ingested episode {i+1}/{len(documents)}: {doc.title[:50]}...")
                
            except Exception as e:
                logger.error(f"❌ Failed to ingest document {i+1}: {e}")
                continue
        
        logger.info(f"✅ Successfully ingested {len(ingested_episodes)} episodes")
        return ingested_episodes
    
    async def test_search(self, search_queries):
        """Test search functionality on ingested data."""
        logger.info("Testing search functionality...")
        
        for query in search_queries:
            try:
                logger.info(f"Searching for: '{query}'")
                results = await self.kg_client.search(query)
                
                logger.info(f"✅ Found {len(results)} results for '{query}'")
                
                # Log top results
                for i, result in enumerate(results[:3]):  # Show top 3 results
                    logger.info(f"  Result {i+1}: {result['fact'][:100]}...")
                
                if not results:
                    logger.info("  No results found")
                
            except Exception as e:
                logger.error(f"❌ Search failed for '{query}': {e}")
    
    async def test_entity_relationships(self, entities):
        """Test entity relationship functionality."""
        logger.info("Testing entity relationship functionality...")
        
        for entity in entities:
            try:
                logger.info(f"Getting relationships for entity: '{entity}'")
                relationships = await self.kg_client.get_related_entities(entity)
                
                logger.info(f"✅ Found relationships for '{entity}'")
                logger.info(f"  Central entity: {relationships.get('central_entity')}")
                logger.info(f"  Related facts: {len(relationships.get('related_facts', []))}")
                
                # Log sample facts
                for i, fact in enumerate(relationships.get('related_facts', [])[:2]):
                    logger.info(f"  Fact {i+1}: {fact['fact'][:100]}...")
                
            except Exception as e:
                logger.error(f"❌ Entity relationship query failed for '{entity}': {e}")
    
    async def run_full_test(self):
        """Run the complete test suite."""
        logger.info("🚀 Starting full KG ingestion test...")
        
        try:
            # Setup
            await self.setup()
            
            # Test connections
            if not await self.test_connection():
                logger.error("❌ Connection test failed, aborting")
                return
            
            # Fetch data
            documents = await self.fetch_intercom_data(limit=5)
            
            if not documents:
                logger.error("❌ No documents to ingest, aborting")
                return
            
            # Ingest documents
            episode_ids = await self.ingest_documents(documents)
            
            if not episode_ids:
                logger.error("❌ No episodes ingested, aborting")
                return
            
            # Test search
            search_queries = [
                "customer support",
                "product features",
                "troubleshooting",
                "getting started"
            ]
            await self.test_search(search_queries)
            
            # Test entity relationships
            entities = ["customer", "product", "support", "analytics"]
            await self.test_entity_relationships(entities)
            
            logger.info("✅ Full test completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Test failed with error: {e}")
            raise
        finally:
            await self.teardown()


async def main():
    """Main function to run the test."""
    print("=" * 60)
    print("KG INGESTION TEST - Intercom to Neo4j")
    print("=" * 60)
    
    # Check environment variables
    required_env_vars = [
        "NEO4J_PASSWORD",
        "LLM_API_KEY",
        "EMBEDDING_API_KEY"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        logger.info("Please set these variables in your .env file")
        return
    
    # Optional Intercom token check
    if not os.getenv("INTERCOM_ACCESS_TOKEN"):
        logger.warning("⚠️ INTERCOM_ACCESS_TOKEN not set, will use mock data")
    
    # Run test
    test = KGIngestionTest()
    await test.run_full_test()


if __name__ == "__main__":
    asyncio.run(main())
