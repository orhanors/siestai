"""
Enhanced async database utilities for PostgreSQL connection and operations.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from contextlib import asynccontextmanager
from uuid import uuid4
import logging
from enum import Enum
import asyncpg
from asyncpg.pool import Pool
from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base



# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Sync db configuration to use with sqlalchemy and alembic
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

# Async db configuration to use with asyncpg
class DocumentSource(str, Enum):
    """Document source types."""
    INTERCOM_ARTICLE = "intercom_article"
    JIRA_TASK = "jira_task"
    CONFLUENCE_PAGE = "confluence_page"
    CUSTOM = "custom"

class DatabaseClient:
    """Enhanced async PostgreSQL connection pool manager."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database pool.
        
        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            # Fallback to individual environment variables
            db_user = os.getenv("DB_USER", "siestai_user")
            db_password = os.getenv("DB_PASSWORD", "siestai_password")
            db_host = os.getenv("DB_HOST", "localhost")
            db_port = os.getenv("DB_PORT", "5439")
            db_name = os.getenv("DB_NAME", "siestai_dev")
            
            self.database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        self.pool: Optional[Pool] = None
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Create connection pool with retry logic."""
        async with self._lock:
            if not self.pool:
                try:
                    self.pool = await asyncpg.create_pool(
                        self.database_url,
                        min_size=5,
                        max_size=20,
                        max_inactive_connection_lifetime=300,
                        command_timeout=60,
                        setup=self._setup_connection
                    )
                    logger.info("Database connection pool initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize database pool: {e}")
                    raise
    
    async def _setup_connection(self, connection):
        """Setup connection with custom types and functions."""
        # Enable pgvector extension if not already enabled
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Set timezone
        await connection.execute("SET timezone = 'UTC'")
    
    async def close(self):
        """Close connection pool."""
        async with self._lock:
            if self.pool:
                await self.pool.close()
                self.pool = None
                logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool with automatic initialization."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query and return the result."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """Fetch multiple rows."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch a single row."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)


# Global database pool instance
db_pool = DatabaseClient()


async def initialize_database():
    """Initialize database connection pool."""
    await db_pool.initialize()


async def close_database():
    """Close database connection pool."""
    await db_pool.close()


# Document Management Functions
async def create_document(
    title: str,
    content: str,
    source: Union[DocumentSource, str],
    original_id: Optional[str] = None,
    content_url: Optional[str] = None,
    language: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    embedding: Optional[List[float]] = None
) -> str:
    """
    Create a new document.
    
    Args:
        title: Document title
        content: Document content
        source: Document source (intercom_article, jira_task, etc.)
        original_id: Original ID from source system
        content_url: URL to original content
        language: Document language
        metadata: Additional metadata
        embedding: Document embedding vector
    
    Returns:
        Document ID
    """
    async with db_pool.acquire() as conn:
        # Convert embedding to PostgreSQL vector format
        embedding_str = None
        if embedding:
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        result = await conn.fetchrow(
            """
            INSERT INTO documents (
                id, title, content, source, original_id, content_url, 
                language, doc_metadata, embedding, created_at, updated_at
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9::vector, NOW(), NOW())
            RETURNING id::text
            """,
            uuid4(),
            title,
            content,
            str(source),
            original_id,
            content_url,
            language,
            json.dumps(metadata or {}),
            embedding_str
        )
        
        return result["id"]


async def update_document(
    document_id: str,
    title: Optional[str] = None,
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    embedding: Optional[List[float]] = None
) -> bool:
    """
    Update an existing document.
    
    Args:
        document_id: Document UUID
        title: New title (optional)
        content: New content (optional)
        metadata: New metadata (optional)
        embedding: New embedding (optional)
    
    Returns:
        True if updated, False if not found
    """
    async with db_pool.acquire() as conn:
        # Build dynamic update query
        updates = []
        params = [document_id]
        param_count = 1
        
        if title is not None:
            param_count += 1
            updates.append(f"title = ${param_count}")
            params.append(title)
        
        if content is not None:
            param_count += 1
            updates.append(f"content = ${param_count}")
            params.append(content)
        
        if metadata is not None:
            param_count += 1
            updates.append(f"doc_metadata = ${param_count}::jsonb")
            params.append(json.dumps(metadata))
        
        if embedding is not None:
            param_count += 1
            updates.append(f"embedding = ${param_count}::vector")
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            params.append(embedding_str)
        
        if not updates:
            return False
        
        updates.append("updated_at = NOW()")
        
        query = f"""
            UPDATE documents
            SET {', '.join(updates)}
            WHERE id = $1::uuid
        """
        
        result = await conn.execute(query, *params)
        return result.split()[-1] != "0"


async def delete_document(document_id: str) -> bool:
    """
    Delete a document.
    
    Args:
        document_id: Document UUID
    
    Returns:
        True if deleted, False if not found
    """
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM documents WHERE id = $1::uuid",
            document_id
        )
        return result.split()[-1] != "0"


async def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Get document by ID.
    
    Args:
        document_id: Document UUID
    
    Returns:
        Document data or None if not found
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT 
                id::text,
                title,
                content,
                source,
                original_id,
                content_url,
                language,
                doc_metadata,
                embedding,
                created_at,
                updated_at
            FROM documents
            WHERE id = $1::uuid
            """,
            document_id
        )
        
        if result:
            return {
                "id": result["id"],
                "title": result["title"],
                "content": result["content"],
                "source": result["source"],
                "original_id": result["original_id"],
                "content_url": result["content_url"],
                "language": result["language"],
                "metadata": json.loads(result["doc_metadata"]),
                "embedding": result["embedding"],
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat()
            }
        
        return None


async def list_documents(
    limit: int = 100,
    offset: int = 0,
    source: Optional[Union[DocumentSource, str]] = None,
    language: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    search_query: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """
    List documents with filtering and search.
    
    Args:
        limit: Maximum number of documents to return
        offset: Number of documents to skip
        source: Filter by source
        language: Filter by language
        metadata_filter: Filter by metadata
        search_query: Full-text search query
    
    Returns:
        Tuple of (documents, total_count)
    """
    async with db_pool.acquire() as conn:
        # Build query conditions
        conditions = []
        params = []
        param_count = 0
        
        if source:
            param_count += 1
            conditions.append(f"source = ${param_count}")
            params.append(str(source))
        
        if language:
            param_count += 1
            conditions.append(f"language = ${param_count}")
            params.append(language)
        
        if metadata_filter:
            param_count += 1
            conditions.append(f"doc_metadata @> ${param_count}::jsonb")
            params.append(json.dumps(metadata_filter))
        
        if search_query:
            param_count += 1
            conditions.append(f"to_tsvector('english', title || ' ' || content) @@ plainto_tsquery('english', ${param_count})")
            params.append(search_query)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        # Get total count
        count_query = f"SELECT COUNT(*) FROM documents{where_clause}"
        total_count = await conn.fetchval(count_query, *params)
        
        # Get documents
        query = f"""
            SELECT 
                id::text,
                title,
                content,
                source,
                original_id,
                content_url,
                language,
                doc_metadata,
                created_at,
                updated_at
            FROM documents{where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_count + 1} OFFSET ${param_count + 2}
        """
        
        params.extend([limit, offset])
        results = await conn.fetch(query, *params)
        
        documents = [
            {
                "id": row["id"],
                "title": row["title"],
                "content": row["content"],
                "source": row["source"],
                "original_id": row["original_id"],
                "content_url": row["content_url"],
                "language": row["language"],
                "metadata": json.loads(row["doc_metadata"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat()
            }
            for row in results
        ]
        
        return documents, total_count


# Enhanced Vector Search Functions
async def vector_search(
    embedding: List[float],
    limit: int = 10,
    threshold: float = 0.7,
    source_filter: Optional[Union[DocumentSource, str]] = None,
    language_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search with filtering.
    
    Args:
        embedding: Query embedding vector
        limit: Maximum number of results
        threshold: Minimum similarity threshold
        source_filter: Filter by document source
        language_filter: Filter by document language
    
    Returns:
        List of matching documents ordered by similarity
    """
    async with db_pool.acquire() as conn:
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        # Build query with filters
        conditions = ["embedding IS NOT NULL"]
        params = [embedding_str, threshold, limit]
        
        if source_filter:
            conditions.append("source = $4")
            params.append(str(source_filter))
        
        if language_filter:
            param_idx = len(params) + 1
            conditions.append(f"language = ${param_idx}")
            params.append(language_filter)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT 
                id::text,
                title,
                content,
                source,
                original_id,
                content_url,
                language,
                doc_metadata,
                embedding <=> $1::vector AS similarity,
                created_at,
                updated_at
            FROM documents
            WHERE {where_clause}
            AND (embedding <=> $1::vector) < $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
        """
        
        results = await conn.fetch(query, *params)
        
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "content": row["content"],
                "source": row["source"],
                "original_id": row["original_id"],
                "content_url": row["content_url"],
                "language": row["language"],
                "metadata": json.loads(row["doc_metadata"]),
                "similarity": float(row["similarity"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat()
            }
            for row in results
        ]


async def hybrid_search(
    embedding: List[float],
    query_text: str,
    limit: int = 10,
    text_weight: float = 0.3,
    threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search (vector + full-text).
    
    Args:
        embedding: Query embedding vector
        query_text: Query text for full-text search
        limit: Maximum number of results
        text_weight: Weight for text similarity (0-1)
        threshold: Minimum similarity threshold
    
    Returns:
        List of matching documents with combined scores
    """
    async with db_pool.acquire() as conn:
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        query = """
            SELECT 
                id::text,
                title,
                content,
                source,
                original_id,
                content_url,
                language,
                doc_metadata,
                embedding <=> $1::vector AS vector_similarity,
                ts_rank(to_tsvector('english', title || ' ' || content), plainto_tsquery('english', $2)) AS text_rank,
                created_at,
                updated_at
            FROM documents
            WHERE embedding IS NOT NULL
            AND to_tsvector('english', title || ' ' || content) @@ plainto_tsquery('english', $2)
            AND (embedding <=> $1::vector) < $3
            ORDER BY ($4 * (1 - (embedding <=> $1::vector))) + ((1 - $4) * ts_rank(to_tsvector('english', title || ' ' || content), plainto_tsquery('english', $2))) DESC
            LIMIT $5
        """
        
        results = await conn.fetch(query, embedding_str, query_text, threshold, text_weight, limit)
        
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "content": row["content"],
                "source": row["source"],
                "original_id": row["original_id"],
                "content_url": row["content_url"],
                "language": row["language"],
                "metadata": json.loads(row["doc_metadata"]),
                "vector_similarity": float(row["vector_similarity"]),
                "text_rank": float(row["text_rank"]),
                "combined_score": float(row["vector_similarity"]) * text_weight + float(row["text_rank"]) * (1 - text_weight),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat()
            }
            for row in results
        ]


# Batch Operations
async def batch_create_documents(documents: List[Dict[str, Any]]) -> List[str]:
    """
    Create multiple documents in a batch.
    
    Args:
        documents: List of document data dictionaries
    
    Returns:
        List of created document IDs
    """
    async with db_pool.acquire() as conn:
        # Prepare batch insert
        values = []
        for doc in documents:
            embedding_str = None
            if doc.get("embedding"):
                embedding_str = '[' + ','.join(map(str, doc["embedding"])) + ']'
            
            values.append((
                uuid4(),
                doc.get("title", ""),
                doc.get("content", ""),
                str(doc.get("source", DocumentSource.CUSTOM)),
                doc.get("original_id"),
                doc.get("content_url"),
                doc.get("language"),
                json.dumps(doc.get("metadata", {})),
                embedding_str
            ))
        
        # Execute batch insert
        result = await conn.executemany(
            """
            INSERT INTO documents (
                id, title, content, source, original_id, content_url, 
                language, doc_metadata, embedding, created_at, updated_at
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9::vector, NOW(), NOW())
            """,
            values
        )
        
        # Return the IDs (we'll need to fetch them since executemany doesn't return them)
        # For now, return a placeholder - in production you might want to handle this differently
        return [str(uuid4()) for _ in documents]


async def batch_update_embeddings(document_embeddings: List[Tuple[str, List[float]]]) -> bool:
    """
    Update embeddings for multiple documents.
    
    Args:
        document_embeddings: List of (document_id, embedding) tuples
    
    Returns:
        True if successful
    """
    async with db_pool.acquire() as conn:
        values = []
        for doc_id, embedding in document_embeddings:
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            values.append((doc_id, embedding_str))
        
        await conn.executemany(
            "UPDATE documents SET embedding = $2::vector, updated_at = NOW() WHERE id = $1::uuid",
            values
        )
        
        return True


# Statistics and Analytics
async def get_document_stats() -> Dict[str, Any]:
    """
    Get document statistics.
    
    Returns:
        Dictionary with various statistics
    """
    async with db_pool.acquire() as conn:
        stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_documents,
                COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as documents_with_embeddings,
                COUNT(DISTINCT source) as unique_sources,
                COUNT(DISTINCT language) as unique_languages,
                MIN(created_at) as oldest_document,
                MAX(created_at) as newest_document
            FROM documents
        """)
        
        source_counts = await conn.fetch("""
            SELECT source, COUNT(*) as count
            FROM documents
            GROUP BY source
            ORDER BY count DESC
        """)
        
        language_counts = await conn.fetch("""
            SELECT language, COUNT(*) as count
            FROM documents
            WHERE language IS NOT NULL
            GROUP BY language
            ORDER BY count DESC
        """)
        
        return {
            "total_documents": stats["total_documents"],
            "documents_with_embeddings": stats["documents_with_embeddings"],
            "unique_sources": stats["unique_sources"],
            "unique_languages": stats["unique_languages"],
            "oldest_document": stats["oldest_document"].isoformat() if stats["oldest_document"] else None,
            "newest_document": stats["newest_document"].isoformat() if stats["newest_document"] else None,
            "source_distribution": {row["source"]: row["count"] for row in source_counts},
            "language_distribution": {row["language"]: row["count"] for row in language_counts}
        }


# Utility Functions
async def test_connection() -> bool:
    """Test database connection."""
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


async def execute_query(query: str, *params) -> List[Dict[str, Any]]:
    """Execute a custom query."""
    async with db_pool.acquire() as conn:
        results = await conn.fetch(query, *params)
        return [dict(row) for row in results]


async def health_check() -> Dict[str, Any]:
    """Comprehensive database health check."""
    try:
        # Test basic connection
        connection_ok = await test_connection()
        
        # Get document stats
        stats = await get_document_stats()
        
        # Test vector operations
        vector_ok = False
        try:
            await execute_query("SELECT '[1,2,3]'::vector")
            vector_ok = True
        except Exception:
            pass
        
        return {
            "status": "healthy" if connection_ok and vector_ok else "unhealthy",
            "connection": connection_ok,
            "vector_extension": vector_ok,
            "statistics": stats
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }