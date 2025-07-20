from faststream.nats import NatsBroker
from faststream import Logger
from faststream.nats import NatsMessage
from pydantic import BaseModel, Field
import os
import asyncio
import nats
from nats.js.api import ConsumerConfig
import json
from app.dto.memory_ingest_dto import MemoryIngestDto
from app.utils.logger import SiestaiLogger

# Initialize logger
logger = SiestaiLogger("siestai.database.ingest")

broker = NatsBroker("nats://localhost:4222")

DB_INGEST_SUBJECT = os.getenv("INGEST_DB_SUBJECT", "siestai.v1.ingest.database.*")
STREAM_NAME = os.getenv("INGEST_STREAM_NAME", "SIESTAI-V1-MEMORY-INGEST")
CONSUMER_NAME = os.getenv("INGEST_CONSUMER_NAME", "siestai-database-ingest-job")
QUEUE_GROUP = "db-ingest-group"

async def create_jetstream_consumer():
    """Create JetStream consumer if it doesn't exist."""
    try:
        logger.info("Connecting to NATS JetStream to create consumer")
        
        # Connect to NATS
        nc = await nats.connect("nats://localhost:4222")
        js = nc.jetstream()
        
        logger.debug(f"Checking if consumer '{CONSUMER_NAME}' exists in stream '{STREAM_NAME}'")
        
        # Check if consumer exists
        try:
            await js.consumer_info(STREAM_NAME, CONSUMER_NAME)
            logger.success(f"Consumer '{CONSUMER_NAME}' already exists in stream '{STREAM_NAME}'")
        except Exception as consumer_error:
            logger.info(f"Consumer '{CONSUMER_NAME}' does not exist, creating new consumer")
            
            # Create consumer with minimal configuration
            consumer_config = ConsumerConfig(
                durable_name=CONSUMER_NAME,
                filter_subject=DB_INGEST_SUBJECT,
                ack_policy="explicit",
                deliver_group=QUEUE_GROUP,
            )
            
            # Create consumer
            await js.add_consumer(STREAM_NAME, consumer_config)
            logger.success(f"Created JetStream consumer '{CONSUMER_NAME}' in stream '{STREAM_NAME}'")
            logger.debug(f"Consumer configuration: filter_subject={DB_INGEST_SUBJECT}, queue_group={QUEUE_GROUP}")
        
        await nc.close()
        logger.debug("NATS connection closed")
        
    except Exception as e:
        logger.error(f"Failed to create JetStream consumer: {e}")
        logger.debug(f"Consumer creation failed with stream='{STREAM_NAME}', consumer='{CONSUMER_NAME}', subject='{DB_INGEST_SUBJECT}'")
        raise

async def ingest_to_database(memory_ingest_dto: MemoryIngestDto):
    """Ingest a batch of data to the database."""
    logger.database(f"Starting database ingestion for source: {memory_ingest_dto.source}")
    logger.debug(f"Ingestion details: source={memory_ingest_dto.source}, data_type={getattr(memory_ingest_dto, 'data_type', 'unknown')}")
    
    try:
        # TODO: Implement actual database ingestion logic
        logger.info(f"Database ingestion completed for source: {memory_ingest_dto.source}")
        return True
    except Exception as e:
        logger.error(f"Database ingestion failed for source {memory_ingest_dto.source}: {e}")
        raise

async def process_database_ingest():
    """Process messages from JetStream consumer."""
    logger.info("Starting database ingest message processing")
    logger.debug(f"Processing configuration: stream='{STREAM_NAME}', consumer='{CONSUMER_NAME}', subject='{DB_INGEST_SUBJECT}'")
    
    try:
        # Connect to NATS
        logger.debug("Connecting to NATS JetStream")
        nc = await nats.connect("nats://localhost:4222")
        js = nc.jetstream()
        
        # Subscribe to the consumer
        logger.debug(f"Subscribing to consumer '{CONSUMER_NAME}' in stream '{STREAM_NAME}'")
        subscription = await js.pull_subscribe(
            subject=DB_INGEST_SUBJECT,
            durable=CONSUMER_NAME,
            stream=STREAM_NAME
        )
        
        logger.success("JetStream subscription established, starting message processing loop")
        
        while True:
            try:
                # Pull messages
                messages = await subscription.fetch(batch=1, timeout=1)
                
                for msg in messages:
                    try:
                        # Parse the message
                        data = json.loads(msg.data.decode())
                        logger.debug(f"Received message with data: {data}")

                        task = MemoryIngestDto(**data)
                        
                        # Get delivery count from headers
                        delivery_count = msg.header.get("Nats-Msg-Delivery-Count", 0) if msg.header else 0
                        max_retries = 20
                        
                        logger.info(f"Processing task from source '{task.source}' (attempt {delivery_count + 1}/{max_retries})")
                        
                        # Ingest the data to the database
                        await ingest_to_database(task)
                        
                        logger.success(f"Task from source '{task.source}' completed successfully")
                        
                        # Acknowledge the message
                        await msg.ack()
                        logger.debug(f"Message from source '{task.source}' acknowledged")
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse message JSON: {e}")
                        logger.debug(f"Raw message data: {msg.data}")
                        await msg.nak()
                        logger.debug("Message negatively acknowledged due to JSON parsing error")
                        
                    except Exception as e:
                        logger.error(f"Error processing message from source '{getattr(task, 'source', 'unknown')}': {e}")
                        logger.debug(f"Message processing failed, delivery count: {delivery_count + 1}")
                        
                        # Negative acknowledgment for failed messages
                        await msg.nak()
                        logger.debug("Message negatively acknowledged (will retry)")
                
            except Exception as e:
                # No messages available or other non-critical errors
                logger.debug(f"No messages available or minor error: {e}")
                pass
                
    except Exception as e:
        logger.error(f"Critical error in JetStream processing: {e}")
        logger.debug(f"JetStream processing failed with stream='{STREAM_NAME}', consumer='{CONSUMER_NAME}'")
        raise
    finally:
        try:
            await nc.close()
            logger.debug("NATS connection closed")
        except Exception as e:
            logger.warning(f"Error closing NATS connection: {e}")
