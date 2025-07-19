#!/usr/bin/env python3
"""
Startup script for Siestai Database Ingest Job
Run with: python start_database_ingest.py
Or with poetry: poetry run database-ingest
"""

import os
import asyncio
from pathlib import Path
from app.memory.database.database_ingest_job import broker, STREAM_NAME, DB_INGEST_SUBJECT

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
    
    # Create push consumer manually
    async def create_push_consumer():
        try:
            js = broker._connection.jetstream()
            
            # Check if consumer already exists
            try:
                consumer_info = await js.consumer_info(STREAM_NAME, "siestai-database-ingest-job")
                print(f"✅ Consumer 'siestai-database-ingest-job' already exists")
                
                # Update consumer to push mode
                print(f"🔄 Updating consumer to push mode...")
                consumer_config = {
                    "durable_name": "siestai-database-ingest-job",
                    "filter_subject": DB_INGEST_SUBJECT,
                    "ack_policy": "explicit",
                    "deliver_policy": "all",
                    "deliver_group": "db-ingest-group"
                }
                await js.update_consumer(STREAM_NAME, "siestai-database-ingest-job", **consumer_config)
                print(f"✅ Consumer updated to push mode")
                
            except Exception:
                # Create new push consumer
                print(f"🔧 Creating push consumer 'siestai-database-ingest-job'...")
                consumer_config = {
                    "durable_name": "siestai-database-ingest-job",
                    "filter_subject": DB_INGEST_SUBJECT,
                    "ack_policy": "explicit",
                    "deliver_policy": "all",
                    "deliver_group": "db-ingest-group"
                }
                await js.add_consumer(STREAM_NAME, **consumer_config)
                print(f"✅ Push consumer 'siestai-database-ingest-job' created successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not create push consumer: {e}")
            print("Continuing with regular NATS subscription...")
    
    print("ℹ️  Creating push consumer for automatic message delivery")
    print("🔄 Max retries: 20 attempts before termination")
    
    async def run():
        await broker.start()
        await create_push_consumer()
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