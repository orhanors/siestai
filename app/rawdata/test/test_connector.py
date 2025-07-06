from app.rawdata.document_fetcher import DocumentFetcher
from app.types.document_types import DocumentSource
import asyncio

fetcher = DocumentFetcher()

fetcher.register_connector(DocumentSource.INTERCOM_ARTICLE)

result = asyncio.run(fetcher.fetch_from_source(DocumentSource.INTERCOM_ARTICLE, limit=10))

print(result)