#!/usr/bin/env python3
"""
Startup script for Siestai Database Ingest Job
Run with: python start_database_ingest.py
Or with poetry: poetry run database-ingest
"""

import os
import asyncio
from pathlib import Path
from app.memory.database.database_ingest_job import broker

def main():
    """Start the Database Ingest Job."""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Set environment variables if not already set
    os.environ.setdefault("INGEST_DB_SUBJECT", "siestai.v1.ingest.database.*")
    
    # Configuration
    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    log_level = os.getenv("LOG_LEVEL", "info")
    
    print(f"🚀 Starting Siestai Database Ingest Job...")
    print(f"🔌 NATS URL: {nats_url}")
    print(f"📝 Log Level: {log_level}")
    print(f"📁 Project Root: {project_root}")
    print(f"📨 Subject: {os.getenv('INGEST_DB_SUBJECT')}")
    print("-" * 50)
    
    # The broker is already configured with subscribers in the module
    
    async def run():
        await broker.start()
        print("✅ Database Ingest Job started successfully")
        print("🔄 Listening for messages...")
        try:
            # Keep the application running
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down Database Ingest Job...")
        finally:
            await broker.close()
    
    # Run the application
    asyncio.run(run())

if __name__ == "__main__":
    main() 