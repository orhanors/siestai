"""
Main application entry point using the structured bootstrap system.
"""

import asyncio
from bootstrap import app_bootstrap, startup_hook, shutdown_hook, AppConfig
from app.memory.database.database import create_document, vector_search, list_documents, DocumentSource, db_pool
from app.services.nats.nats_client import NatsService, NatsStreamConfig
import os
# Example startup hooks using decorators
@startup_hook
async def prepare_nats_streams():
    """Prepare NATS streams"""
    # await app_bootstrap.initialize_nats()
    

@shutdown_hook
async def cleanup_background_tasks():
    """Cleanup background tasks"""
    print("🧹 Cleaning up background tasks...")
    # Stop periodic tasks, cleanup resources


# Main application logic
async def run_application_logic():
    """Main application business logic"""
    print("🚀 Running application logic...")
    
    # Create document
    doc_id = await create_document(
        title="My Document",
        content="Document content here",
        source=DocumentSource.INTERCOM_ARTICLE,
        metadata={"author": "John Doe"}
    )
    
    # Search documents
    results = await vector_search(
        embedding=[0.1, 0.2, 0.3] * 512,
        limit=10,
        source_filter=DocumentSource.INTERCOM_ARTICLE
    )
    
    # List documents using context manager
    async with db_pool.acquire() as conn:
        documents, total = await list_documents(limit=100)
        print(f"Found {total} documents")
        
    # Example of using NATS service
    nats_service = app_bootstrap.get_nats_service()
    if nats_service:
        await nats_service.publish_event(os.getenv("INGEST_PGVECTOR_SUBJECT"), {
            "document_id": str(doc_id),
            "title": "My Document"
        })
        print("📨 Event published to NATS")


async def main():
    """Main entry point with structured bootstrap"""
    config = AppConfig(
        debug=True,
        nats_url="nats://localhost:4222",
        log_level="INFO"
    )
    
    # Initialize bootstrap with custom config
    app_bootstrap.config = config
    
    # Use the lifecycle context manager
    async with app_bootstrap.lifespan() as app:
        await run_application_logic()


if __name__ == "__main__":
    asyncio.run(main()) 