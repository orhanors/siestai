#!/usr/bin/env python3
"""
Startup script for Siestai Database Ingest Job
Run with: python start_database_ingest.py
Or with poetry: poetry run database-ingest
"""

import os
import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from app.memory.database.database_ingest_job import (
    STREAM_NAME, 
    DB_INGEST_SUBJECT, 
    create_jetstream_consumer, 
    ingest_memory_to_database
)
from app.utils.logger import setup_logging, SiestaiLogger
from app.config.logging_config import get_logging_config


class DatabaseIngestService:
    """Service class for managing the database ingest job."""
    
    def __init__(self):
        self.logger: Optional[SiestaiLogger] = None
        self.stop_event = asyncio.Event()
        self._setup_environment()
        self._setup_logging()
        self._setup_signal_handlers()
    
    def _setup_environment(self):
        """Setup environment variables and configuration."""
        # Get the project root directory
        self.project_root = Path(__file__).parent
        
        # Set environment variables if not already set
        os.environ.setdefault("INGEST_DB_SUBJECT", "siestai.v1.ingest.database.*")
        
        # Configuration
        self.nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
        self.log_level = os.getenv("LOG_LEVEL", "info")
        self.environment = os.getenv("ENVIRONMENT", "development")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging_config = get_logging_config(self.environment)
        self.logger = setup_logging(
            level=logging_config["level"],
            enable_console=logging_config["console_enabled"]
        )
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
            self.stop_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _print_startup_info(self):
        """Print startup information."""
        self.logger.info("🚀 Starting Siestai Database Ingest Job...")
        self.logger.info(f"🔌 NATS URL: {self.nats_url}")
        self.logger.info(f"📝 Log Level: {self.log_level}")
        self.logger.info(f"🌍 Environment: {self.environment}")
        self.logger.info(f"📁 Project Root: {self.project_root}")
        self.logger.info(f"📨 Subject: {os.getenv('INGEST_DB_SUBJECT')}")
        self.logger.info("-" * 50)
        self.logger.info("ℹ️  Starting JetStream consumer with queue group")
        self.logger.info("🔄 Max retries: 20 attempts before termination")
        self.logger.info("🗑️ Messages will be removed after acknowledgment (WORK_QUEUE retention)")
    
    @asynccontextmanager
    async def _manage_nats_connection(self):
        """Context manager for NATS connection management."""
        try:
            yield
        except Exception as e:
            self.logger.error(f"❌ NATS connection error: {e}")
            raise
        finally:
            self.logger.info("🔌 NATS connection closed")
    
    async def _create_consumer(self):
        """Create JetStream consumer with error handling."""
        try:
            self.logger.info("🔧 Creating JetStream consumer...")
            await create_jetstream_consumer()
            self.logger.success("✅ JetStream consumer created successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to create JetStream consumer: {e}")
            raise
    
    async def _start_message_processing(self):
        """Start message processing with graceful shutdown."""
        try:
            self.logger.info("🔄 Starting message processing...")
            await ingest_memory_to_database()
        except asyncio.CancelledError:
            self.logger.info("🛑 Message processing cancelled")
        except Exception as e:
            self.logger.error(f"❌ Error in message processing: {e}")
            raise
    
    async def run(self):
        """Main run method with comprehensive error handling."""
        try:
            self._print_startup_info()
            
            # Create JetStream consumer
            await self._create_consumer()
            
            # Start message processing
            self.logger.success("✅ Database Ingest Job started successfully")
            self.logger.info("🔄 Listening for messages...")
            
            # Wait for stop signal
            await self.stop_event.wait()
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"❌ Fatal error: {e}")
            sys.exit(1)
        finally:
            self.logger.success("✅ Database Ingest Job shutdown complete")


def main():
    """Main entry point for the Database Ingest Job."""
    try:
        service = DatabaseIngestService()
        asyncio.run(service.run())
    except Exception as e:
        print(f"❌ Failed to start Database Ingest Job: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 