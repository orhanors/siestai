"""
Centralized models file for Alembic migrations.
This file imports and exports all SQLAlchemy models to ensure they are available
for Alembic's autogenerate functionality.

When you add new models to the app, make sure to:
1. Create the model file in app/models/
2. Import it here
3. Add it to the __all__ list
4. Update the models/__init__.py file

This ensures all models are registered with the Base metadata
when this module is imported by Alembic.
"""

# Import all models here
from .documents import Document

# Export all models for Alembic
__all__ = [
    'Document',
]

# Future models can be added here:
# from .users import User
# from .conversations import Conversation
# from .embeddings import Embedding
# 
# __all__ = [
#     'Document',
#     'User',
#     'Conversation', 
#     'Embedding',
# ]

# This ensures all models are registered with the Base metadata
# when this module is imported by Alembic
