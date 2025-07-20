from pydantic import BaseModel, Field
from app.types.document_types import DocumentSource, Credentials

class MemoryIngestDto(BaseModel):
    source: DocumentSource = Field(..., description="The source to fetch all the data from")
    credentials: Credentials = Field(..., description="The credentials to fetch the data from the source")