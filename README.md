# siestai
SiestaI - an intelligent workplace wellness agent that guides employees through effective rest breaks and strategic napping for peak performance.

## Database Migrations

To set up the database and run migrations, follow these steps:

1. **Enable the pgvector extension** (only needed once):
   
   If you are using Docker Compose (as in this project), run:
   
   ```bash
   docker exec siestai_postgres psql -U siestai_user -d siestai_dev -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

2. **Migration Commands**

   **Generate a new migration from model changes:**
   
   ```bash
   alembic revision --autogenerate -m "Description of changes"
   ```

   **Apply all pending migrations:**
   
   ```bash
   alembic upgrade head
   ```

   **Downgrade one migration:**

      ```bash
      alembic downgrade -1
      ```

   **View migration history:**
   
      ```bash
      alembic history
      ```

   **Apply migrations up to a specific revision:**
   
      ```bash
      alembic upgrade <revision_id>
      ```

This will create all necessary tables, including support for vector embeddings.
