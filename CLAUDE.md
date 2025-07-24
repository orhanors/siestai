# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation and Setup
```bash
# Install dependencies
poetry install

# Enable pgvector extension (only needed once)
docker exec siestai_postgres psql -U siestai_user -d siestai_dev -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Database Migrations
```bash
# Generate a new migration from model changes
alembic revision --autogenerate -m "Description of changes"

# Apply all pending migrations
alembic upgrade head

# Downgrade one migration
alembic downgrade -1

# View migration history
alembic history

# Apply migrations up to a specific revision
alembic upgrade <revision_id>
```

### Running Services
The project has multiple entry point scripts defined in pyproject.toml:

```bash
# Start the memory API service
poetry run memory-api

# Start database ingestion service
poetry run database-ingest

# Start knowledge graph ingestion service
poetry run kg-ingest

# Start research agent service
poetry run research-agent
```

Alternatively, run the starter scripts directly:
```bash
python start_memory_api.py
python start_database_ingest.py
python start_kg_ingest.py
python start_research_agent.py
```

### Development
```bash
# Run basic database operations (main.py is a demo/test script)
python main.py

# Run with bootstrap lifecycle management
python main_with_bootstrap.py
```

## Architecture

### Core Components

**Bootstrap System** (`bootstrap.py`): Centralized application lifecycle management with startup/shutdown hooks, handles database initialization, NATS service setup, and logging configuration.

**Memory System**: Multi-layered memory architecture:
- `app/memory/database/`: Vector database operations with PostgreSQL + pgvector
- `app/memory/history/`: Chat session management and conversation history
- `app/memory/knowledge_graph/`: Graph-based knowledge representation using Graphiti

**Services**:
- `app/services/nats/`: Message queue system for inter-service communication
- `app/services/embedding_service.py`: Text embedding generation
- `app/api/memory_api.py`: FastAPI-based memory service endpoints

**Data Connectors** (`app/rawdata/`): Pluggable connectors for external data sources (Intercom, JIRA, etc.)

**Agents** (`app/agents/`): Specialized AI agents including research, database, and crypto agents

### Database Schema
- Uses SQLAlchemy with Alembic migrations
- Vector embeddings stored in PostgreSQL with pgvector extension
- Models defined in `app/models/`: documents, chat history, and core entities

### Messaging Architecture
- NATS-based pub/sub messaging
- Stream configurations in `app/services/nats/stream_configs.py`
- Bootstrap automatically creates required streams on startup

### Logging
- Rich console logging with environment-specific configurations
- File rotation and component-specific log levels
- Configured in `app/config/logging_config.py`

## Key Patterns

**Service Initialization**: Use the bootstrap system for service lifecycle management. Services should register startup/shutdown hooks rather than handling initialization directly.

**Database Operations**: Always use the connection pool through `app/memory/database/database.py`. The system supports both standalone usage and context manager patterns.

**Memory Architecture**: The system maintains three types of memory - vector database for semantic search, session history for conversations, and knowledge graphs for structured relationships.

**Agent Design**: Agents are self-contained modules with their own configuration and test suites, following a consistent pattern in the `app/agents/` directory.