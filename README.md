# siestai
Siestai agent that helps company workers to get their siesta

## Database Migrations

To set up the database and run migrations, follow these steps:

1. **Enable the pgvector extension** (only needed once):
   
   If you are using Docker Compose (as in this project), run:
   
   ```bash
   docker exec siestai_postgres psql -U siestai_user -d siestai_dev -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

2. **Run Alembic migrations:**
   
   ```bash
   alembic upgrade head
   ```

This will create all necessary tables, including support for vector embeddings.
