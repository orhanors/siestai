"""
Shared types and enums for document handling.
"""

from enum import Enum
from pydantic import BaseModel

class DocumentSource(str, Enum):
    """Document source types."""
    INTERCOM_ARTICLE = "intercom_article"
    JIRA_TASK = "jira_task"
    CONFLUENCE_PAGE = "confluence_page"
    CUSTOM = "custom" 

class Credentials(BaseModel):
    api_key: str