"""Models package for the application."""

# Import from the centralized models file
from ._models import *

# Also import individual models for direct access
from .documents import Document

__all__ = [
    'Document',
] 