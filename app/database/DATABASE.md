# Database Documentation

## Overview

The SiestAI database module provides a comprehensive PostgreSQL-based document storage and retrieval system with vector search capabilities. It uses both SQLAlchemy for ORM operations and asyncpg for high-performance async operations.

## Architecture

### Technology Stack
- **PostgreSQL**: Primary database with pgvector extension for vector operations
- **SQLAlchemy**: ORM for schema management and migrations
- **asyncpg**: High-performance async PostgreSQL driver
- **Alembic**: Database migration management
- **pgvector**: PostgreSQL extension for vector similarity search

### Key Components

1. **DatabasePool**: Async connection pool manager
2. **Document Model**: SQLAlchemy model for document storage
3. **Document Management Functions**: CRUD operations for documents
4. **Vector Search Functions**: Semantic search capabilities
5. **Batch Operations**: Efficient bulk operations
6. **Statistics & Analytics**: Database insights and monitoring

## Database Schema

### Documents Table

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    original_id TEXT,                    -- External system ID (e.g., Intercom article ID)
    content_url TEXT,                    -- URL to original content
    source TEXT NOT NULL,                -- Document source (intercom_article, jira_task, etc.)
    title TEXT,                          -- Document title
    content TEXT NOT NULL,               -- Document content
    language TEXT,                       -- Document language
    doc_metadata JSONB,                  -- Additional metadata (tags, authors, etc.)
    embedding VECTOR(1536),              -- Document embedding vector
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Indexes
- Primary key on `id`
- Vector similarity index on `embedding` (automatically created by pgvector)
- Full-text search index on `title` and `content`

## Setup and Configuration

### Environment Variables

```bash
# Database connection
DB_NAME=siestai_dev
DB_USER=siestai_user
DB_PASSWORD=siestai_password
DB_HOST=localhost
DB_PORT=5439

# Or use a single DATABASE_URL
DATABASE_URL=postgresql://siestai_user:siestai_password@localhost:5439/siestai_dev
```

### Database Initialization

```python
from app.database.database import initialize_database, close_database

# Initialize connection pool
await initialize_database()

# Close connections when done
await close_database()
```

### Running Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## Usage Examples

### Basic Document Operations

```python
from app.database.database import (
    create_document, get_document, update_document, delete_document,
    DocumentSource
)

# Create a document
doc_id = await create_document(
    title="Getting Started Guide",
    content="This is a comprehensive guide...",
    source=DocumentSource.INTERCOM_ARTICLE,
    original_id="article_123",
    content_url="https://help.example.com/getting-started",
    language="en",
    metadata={"category": "tutorial", "author": "john_doe"}
)

# Retrieve a document
document = await get_document(doc_id)
print(f"Document: {document['title']}")

# Update a document
success = await update_document(
    doc_id,
    title="Updated Getting Started Guide",
    metadata={"category": "tutorial", "author": "john_doe", "version": "2.0"}
)

# Delete a document
deleted = await delete_document(doc_id)
```

### Vector Search

```python
from app.database.database import vector_search, hybrid_search

# Sample embedding vector (1536 dimensions)
query_embedding = [0.1, 0.2, 0.3, ...]  # Your embedding here

# Vector similarity search
similar_docs = await vector_search(
    embedding=query_embedding,
    limit=10,
    threshold=0.7,
    source_filter=DocumentSource.INTERCOM_ARTICLE,
    language_filter="en"
)

# Hybrid search (vector + full-text)
hybrid_results = await hybrid_search(
    embedding=query_embedding,
    query_text="getting started tutorial",
    limit=10,
    text_weight=0.3,
    threshold=0.7
)
```

### Batch Operations

```python
from app.database.database import batch_create_documents, batch_update_embeddings

# Batch create documents
documents = [
    {
        "title": "Document 1",
        "content": "Content 1",
        "source": DocumentSource.INTERCOM_ARTICLE,
        "metadata": {"category": "guide"}
    },
    {
        "title": "Document 2", 
        "content": "Content 2",
        "source": DocumentSource.JIRA_TASK,
        "metadata": {"priority": "high"}
    }
]

doc_ids = await batch_create_documents(documents)

# Batch update embeddings
document_embeddings = [
    (doc_ids[0], [0.1, 0.2, 0.3, ...]),
    (doc_ids[1], [0.4, 0.5, 0.6, ...])
]

await batch_update_embeddings(document_embeddings)
```

### Document Listing and Filtering

```python
from app.database.database import list_documents

# List all documents
documents, total_count = await list_documents(limit=50, offset=0)

# Filter by source and language
intercom_docs, count = await list_documents(
    limit=20,
    source=DocumentSource.INTERCOM_ARTICLE,
    language="en"
)

# Search with metadata filter
tutorial_docs, count = await list_documents(
    limit=10,
    metadata_filter={"category": "tutorial"}
)

# Full-text search
search_results, count = await list_documents(
    limit=10,
    search_query="authentication setup"
)
```

### Statistics and Monitoring

```python
from app.database.database import get_document_stats, health_check

# Get database statistics
stats = await get_document_stats()
print(f"Total documents: {stats['total_documents']}")
print(f"Sources: {stats['source_distribution']}")

# Health check
health = await health_check()
print(f"Database status: {health['status']}")
```

## API Reference

### DatabasePool Class

#### Methods

- `initialize()`: Create connection pool with retry logic
- `close()`: Close connection pool
- `acquire()`: Context manager for acquiring connections
- `execute(query, *args)`: Execute a query
- `fetch(query, *args)`: Fetch multiple rows
- `fetchrow(query, *args)`: Fetch a single row
- `fetchval(query, *args)`: Fetch a single value

### Document Management Functions

#### `create_document(title, content, source, **kwargs)`
Creates a new document.

**Parameters:**
- `title` (str): Document title
- `content` (str): Document content
- `source` (DocumentSource): Document source type
- `original_id` (str, optional): External system ID
- `content_url` (str, optional): URL to original content
- `language` (str, optional): Document language
- `metadata` (dict, optional): Additional metadata
- `embedding` (list[float], optional): Document embedding vector

**Returns:** Document ID (str)

#### `get_document(document_id)`
Retrieves a document by ID.

**Parameters:**
- `document_id` (str): Document UUID

**Returns:** Document data (dict) or None

#### `update_document(document_id, **kwargs)`
Updates an existing document.

**Parameters:**
- `document_id` (str): Document UUID
- `title` (str, optional): New title
- `content` (str, optional): New content
- `metadata` (dict, optional): New metadata
- `embedding` (list[float], optional): New embedding

**Returns:** True if updated, False if not found

#### `delete_document(document_id)`
Deletes a document.

**Parameters:**
- `document_id` (str): Document UUID

**Returns:** True if deleted, False if not found

### Search Functions

#### `vector_search(embedding, limit=10, threshold=0.7, **filters)`
Performs vector similarity search.

**Parameters:**
- `embedding` (list[float]): Query embedding vector
- `limit` (int): Maximum number of results
- `threshold` (float): Minimum similarity threshold
- `source_filter` (DocumentSource, optional): Filter by source
- `language_filter` (str, optional): Filter by language

**Returns:** List of matching documents ordered by similarity

#### `hybrid_search(embedding, query_text, limit=10, text_weight=0.3, threshold=0.7)`
Performs hybrid search combining vector and full-text search.

**Parameters:**
- `embedding` (list[float]): Query embedding vector
- `query_text` (str): Query text for full-text search
- `limit` (int): Maximum number of results
- `text_weight` (float): Weight for text similarity (0-1)
- `threshold` (float): Minimum similarity threshold

**Returns:** List of matching documents with combined scores

### Batch Operations

#### `batch_create_documents(documents)`
Creates multiple documents in a batch.

**Parameters:**
- `documents` (list[dict]): List of document data dictionaries

**Returns:** List of created document IDs

#### `batch_update_embeddings(document_embeddings)`
Updates embeddings for multiple documents.

**Parameters:**
- `document_embeddings` (list[tuple]): List of (document_id, embedding) tuples

**Returns:** True if successful

### Utility Functions

#### `get_document_stats()`
Returns comprehensive document statistics.

**Returns:** Dictionary with statistics including counts, distributions, and date ranges

#### `health_check()`
Performs comprehensive database health check.

**Returns:** Dictionary with health status and component checks

#### `test_connection()`
Tests basic database connectivity.

**Returns:** True if connection successful, False otherwise

## DocumentSource Enum

```python
class DocumentSource(str, Enum):
    INTERCOM_ARTICLE = "intercom_article"
    JIRA_TASK = "jira_task"
    CONFLUENCE_PAGE = "confluence_page"
    CUSTOM = "custom"
```

## Performance Considerations

### Connection Pooling
- Default pool size: 5-20 connections
- Automatic connection lifecycle management
- Connection timeout: 60 seconds

### Vector Operations
- Uses pgvector extension for efficient similarity search
- Supports cosine distance, L2 distance, and inner product
- Automatic vector dimension validation

### Indexing
- Primary key index on `id`
- Vector similarity index on `embedding`
- Full-text search indexes on `title` and `content`
- JSONB indexes on `doc_metadata`

### Batch Operations
- Use batch operations for bulk inserts/updates
- Consider chunking large batches (1000+ documents)
- Monitor memory usage for large embedding vectors

## Error Handling

The database module includes comprehensive error handling:

```python
try:
    document = await get_document("invalid-uuid")
except Exception as e:
    logger.error(f"Failed to retrieve document: {e}")
    # Handle error appropriately
```

### Common Error Scenarios
- **Connection failures**: Automatic retry with exponential backoff
- **Invalid UUIDs**: Proper validation and error messages
- **Vector dimension mismatches**: Automatic validation
- **JSON serialization errors**: Graceful handling of invalid metadata

## Monitoring and Logging

### Logging Configuration
- Database operations are logged at INFO level
- Errors are logged at ERROR level with full context
- Connection pool events are logged for debugging

### Health Monitoring
- Connection pool status
- Vector extension availability
- Document statistics
- Query performance metrics

## Best Practices

### Document Creation
1. Always provide meaningful titles and content
2. Use appropriate source types
3. Include relevant metadata for filtering
4. Validate embedding dimensions before insertion

### Search Optimization
1. Use appropriate similarity thresholds
2. Combine vector and text search for better results
3. Filter by source/language when possible
4. Monitor query performance

### Data Management
1. Regular database backups
2. Monitor storage usage
3. Archive old documents when necessary
4. Validate data integrity periodically

## Troubleshooting

### Common Issues

1. **pgvector extension not found**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. **Connection pool exhaustion**
   - Increase pool size
   - Check for connection leaks
   - Monitor query performance

3. **Vector dimension mismatch**
   - Ensure embedding vectors match table schema
   - Update schema if needed: `ALTER TABLE documents ALTER COLUMN embedding TYPE vector(1536);`

4. **Performance issues**
   - Check query execution plans
   - Ensure proper indexing
   - Monitor connection pool usage

### Debugging Queries

```python
# Enable query logging
import logging
logging.getLogger('asyncpg').setLevel(logging.DEBUG)

# Test specific queries
result = await execute_query("EXPLAIN ANALYZE SELECT * FROM documents WHERE source = $1", "intercom_article")
```

## Migration Guide

### Adding New Document Sources
1. Add new enum value to `DocumentSource`
2. Update documentation
3. Consider adding source-specific indexes if needed

### Schema Changes
1. Create new Alembic migration
2. Test migration on development database
3. Update model definitions
4. Update API documentation

### Performance Tuning
1. Monitor query performance
2. Add appropriate indexes
3. Optimize batch operations
4. Consider partitioning for large datasets
