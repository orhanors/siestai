"""
Example usage of the data connector interface.
"""

import asyncio
from app.rawdata.data_connector_interface import create_connector
from app.types.document_types import DocumentSource, Credentials
from app.memory.database.database import create_document
import os

async def main():
    """Example of using data connectors to fetch and store documents."""
    
    # Create credentials for testing
    credentials = Credentials(api_key=os.getenv("INTERCOM_ACCESS_TOKEN"))
    
    # Create a mock connector for testing
    mock_connector = create_connector(DocumentSource.INTERCOM_ARTICLE)
    
    # Test connection
    if await mock_connector.test_connection(credentials):
        print("✅ Mock connector connection successful")
    else:
        print("❌ Mock connector connection failed")
        return
    
    # Fetch documents
    result = await mock_connector.get_documents_with_validation(credentials)
    print(f"📄 Fetched {len(result.documents)} documents")
    
    # Store documents in database
    for doc_data in result.documents:
        try:
            doc_id = await create_document(
                title=doc_data.title,
                content=doc_data.content,
                source=doc_data.source,
                original_id=doc_data.original_id,
                content_url=doc_data.content_url,
                language=doc_data.language,
                metadata=doc_data.metadata
            )
            print(f"💾 Stored document: {doc_data.title} (ID: {doc_id})")
        except Exception as e:
            print(f"❌ Failed to store document {doc_data.title}: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 