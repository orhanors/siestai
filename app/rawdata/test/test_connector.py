from app.rawdata.document_fetcher import DocumentFetcher
from app.types.document_types import DocumentSource
import asyncio

fetcher = DocumentFetcher()

fetcher.register_connector(DocumentSource.INTERCOM_ARTICLE)
fetcher.register_connector(DocumentSource.CUSTOM)

result = asyncio.run(fetcher.fetch_from_all_sources(limit=100))

