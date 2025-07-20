#!/usr/bin/env python3
"""
Startup script for Siestai Database Ingest Job
Run with: python start_database_ingest.py
Or with poetry: poetry run database-ingest
"""

import os
import asyncio
import signal
from pathlib import Path
from app.memory.database.database_ingest_job import STREAM_NAME, DB_INGEST_SUBJECT, create_jetstream_consumer, ingest_memory_to_database

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
    
    print("ℹ️  Starting JetStream consumer with queue group")
    print("🔄 Max retries: 20 attempts before termination")
    print("🗑️ Messages will be removed after acknowledgment (WORK_QUEUE retention)")
    
    # Create event to keep process alive
    stop_event = asyncio.Event()
    
    def signal_handler():
        print("\n🛑 Received shutdown signal...")
        stop_event.set()
    
    async def run():
        # Create JetStream consumer first
        print("🔧 Creating JetStream consumer...")
        await create_jetstream_consumer()
        
        # Start JetStream message processing
        print("✅ Database Ingest Job started successfully")
        print("🔄 Listening for messages...")
        
        try:
            # Start message processing
            await ingest_memory_to_database()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down Database Ingest Job...")
        finally:
            print("✅ Database Ingest Job shutdown complete")
    
    # Run the application
    asyncio.run(run())

if __name__ == "__main__":
    main() 