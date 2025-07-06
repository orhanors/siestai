import asyncio
from app.memory.database.database import initialize_database, close_database, create_document, vector_search, list_documents, DocumentSource, db_pool

# Basic usage
async def main():
    await initialize_database()
    print("Database initialized")
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
    
    await with_context()
    await close_database()

# Using context managers
async def with_context():
    async with db_pool.acquire() as conn:
        documents, total = await list_documents(limit=100)
        print(f"Found {total} documents")

if __name__ == "__main__":
    asyncio.run(main())