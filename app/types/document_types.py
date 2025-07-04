"""
Shared types and enums for document handling.
"""

from enum import Enum


class DocumentSource(str, Enum):
    """Document source types."""
    INTERCOM_ARTICLE = "intercom_article"
    JIRA_TASK = "jira_task"
    CONFLUENCE_PAGE = "confluence_page"
    CUSTOM = "custom" 