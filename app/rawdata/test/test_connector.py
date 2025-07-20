from app.rawdata.document_fetcher import DocumentFetcher
from app.types.document_types import DocumentSource, Credentials
import asyncio
import os
fetcher = DocumentFetcher()

fetcher.register_connector(DocumentSource.INTERCOM_ARTICLE)
fetcher.register_connector(DocumentSource.CUSTOM)

# Create test credentials
credentials = Credentials(api_key=os.getenv("INTERCOM_ACCESS_TOKEN"))

result = asyncio.run(fetcher.fetch_from_source(DocumentSource.INTERCOM_ARTICLE, credentials, limit=25))

