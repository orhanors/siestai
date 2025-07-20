from faststream.nats import NatsBroker
from faststream import Logger
from faststream.nats import NatsMessage
from pydantic import BaseModel, Field
import os
import asyncio
import nats
from nats.js.api import ConsumerConfig
import json
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from app.utils.logger import setup_logging, SiestaiLogger

class Task(BaseModel):
    id: int = Field(..., example=1)
    payload: str = Field(..., example="do something")

# Configuration
NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
DB_INGEST_SUBJECT = os.getenv("INGEST_DB_SUBJECT", "siestai.v1.ingest.database.*")
STREAM_NAME = os.getenv("INGEST_STREAM_NAME", "SIESTAI-V1-MEMORY-INGEST")
CONSUMER_NAME = os.getenv("INGEST_CONSUMER_NAME", "siestai-database-ingest-job")
QUEUE_GROUP = "db-ingest-group"
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "20"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "1"))

# Initialize broker
broker = NatsBroker(NATS_URL)

# Setup logger
logger = setup_logging()


@asynccontextmanager
async def get_nats_connection():
    """Context manager for NATS connection."""
    nc = None
    try:
        nc = await nats.connect(NATS_URL)
        logger.debug(f"🔌 Connected to NATS at {NATS_URL}")
        yield nc
    except Exception as e:
        logger.error(f"❌ Failed to connect to NATS: {e}")
        raise
    finally:
        if nc:
            await nc.close()
            logger.debug("🔌 NATS connection closed")


async def create_jetstream_consumer():
    """Create JetStream consumer if it doesn't exist."""
    async with get_nats_connection() as nc:
        try:
            js = nc.jetstream()
            
            # Check if consumer exists
            try:
                await js.consumer_info(STREAM_NAME, CONSUMER_NAME)
                logger.info(f"✅ Consumer '{CONSUMER_NAME}' already exists")
                return
            except Exception:
                # Consumer doesn't exist, create it
                pass
            
            # Create consumer with proper configuration
            consumer_config = ConsumerConfig(
                durable_name=CONSUMER_NAME,
                filter_subject=DB_INGEST_SUBJECT,
                ack_policy="explicit",
                deliver_group=QUEUE_GROUP,
                max_deliver=MAX_RETRIES,
                ack_wait=30_000,  # 30 seconds
                backoff=[1_000, 5_000, 10_000, 30_000],  # Progressive backoff
                num_replicas=1,
            )
            
            # Create consumer
            await js.add_consumer(STREAM_NAME, consumer_config)
            logger.success(f"✅ Created JetStream consumer '{CONSUMER_NAME}'")
            
        except Exception as e:
            logger.error(f"❌ Error creating consumer: {e}")
            raise


async def process_message(msg, delivery_count: int) -> bool:
    """
    Process a single message.
    
    Args:
        msg: NATS message object
        delivery_count: Current delivery attempt number
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Parse the message
        data = json.loads(msg.data.decode())
        task = Task(**data)
        
        logger.info(f"Processing task {task.id}: {task.payload} (attempt {delivery_count + 1}/{MAX_RETRIES})")
        
        # TODO: Implement actual database processing logic here
        # For now, simulate some work
        await asyncio.sleep(0.1)
        
        logger.success(f"✅ Task {task.id} completed successfully")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in message: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error processing task: {e}")
        return False


async def ingest_memory_to_database():
    """Process messages from JetStream consumer with improved error handling."""
    async with get_nats_connection() as nc:
        try:
            js = nc.jetstream()
            
            # Subscribe to the consumer
            subscription = await js.pull_subscribe(
                subject=DB_INGEST_SUBJECT,
                durable=CONSUMER_NAME,
                stream=STREAM_NAME
            )
            
            logger.info(f"🔄 Starting JetStream message processing...")
            logger.info(f"📊 Configuration: batch_size={BATCH_SIZE}, timeout={FETCH_TIMEOUT}s, max_retries={MAX_RETRIES}")
            
            consecutive_empty_fetches = 0
            max_empty_fetches = 10  # Log warning after 10 consecutive empty fetches
            
            while True:
                try:
                    # Pull messages
                    messages = await subscription.fetch(batch=BATCH_SIZE, timeout=FETCH_TIMEOUT)
                    
                    if not messages:
                        consecutive_empty_fetches += 1
                        if consecutive_empty_fetches >= max_empty_fetches:
                            logger.debug(f"⏳ No messages available (empty fetches: {consecutive_empty_fetches})")
                            consecutive_empty_fetches = 0  # Reset counter
                        continue
                    
                    # Reset empty fetch counter
                    consecutive_empty_fetches = 0
                    
                    for msg in messages:
                        try:
                            # Get delivery count from headers
                            delivery_count = msg.header.get("Nats-Msg-Delivery-Count", 0) if msg.header else 0
                            
                            # Process the message
                            success = await process_message(msg, delivery_count)
                            
                            if success:
                                # Acknowledge the message
                                await msg.ack()
                                logger.debug(f"📨 Message {msg.data.decode()[:50]}... acknowledged")
                            else:
                                # Negative acknowledgment for failed messages
                                await msg.nak()
                                logger.warning(f"📨 Message negatively acknowledged (will retry)")
                                
                        except Exception as e:
                            logger.error(f"❌ Error handling message: {e}")
                            # Negative acknowledgment for failed messages
                            await msg.nak()
                
                except asyncio.TimeoutError:
                    # This is expected when no messages are available
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in message processing loop: {e}")
                    # Continue processing other messages
                    continue
                
        except Exception as e:
            logger.error(f"❌ Error in JetStream processing: {e}")
            raise


# Legacy function for backward compatibility
async def start_ingest():
    """Legacy function to start the ingest process."""
    await create_jetstream_consumer()
    await ingest_memory_to_database()
