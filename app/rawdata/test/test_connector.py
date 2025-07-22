from app.rawdata.document_fetcher import DocumentFetcher
from app.types.document_types import DocumentSource, Credentials
import asyncio
import os
from dotenv import load_dotenv
from app.dto.document_dto import FetchMetadata

load_dotenv()

fetcher = DocumentFetcher()

fetcher.register_connector(DocumentSource.INTERCOM_ARTICLE)
fetcher.register_connector(DocumentSource.CUSTOM)

# Create test credentials
intercom_access_token = os.getenv("INTERCOM_ACCESS_TOKEN")
credentials = Credentials(api_key=intercom_access_token)

result = asyncio.run(fetcher.fetch_from_source(
    DocumentSource.INTERCOM_ARTICLE,
    credentials,
    FetchMetadata(metadata={"limit": 25})
))
print(result)

