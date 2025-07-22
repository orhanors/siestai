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
from app.utils.logger import get_logger
from app.rawdata.document_fetcher import DocumentFetcher
from app.types.document_types import DocumentSource, Credentials
from app.dto.document_dto import FetchMetadata, PaginatedDocuments
from app.memory.database.database import create_document
from app.utils.rate_limiter import RateLimitConfig

# Initialize logger
logger = get_logger("siestai.database.ingest")

broker = NatsBroker("nats://localhost:4222")

DB_INGEST_SUBJECT = os.getenv("INGEST_DB_SUBJECT", "siestai.v1.ingest.database.*")
STREAM_NAME = os.getenv("INGEST_STREAM_NAME", "SIESTAI-V1-MEMORY-INGEST")
CONSUMER_NAME = os.getenv("INGEST_CONSUMER_NAME", "siestai-database-ingest-job")
QUEUE_GROUP = "db-ingest-group"

def _get_rate_limit_config(source: DocumentSource) -> RateLimitConfig:
    """Get rate limit configuration for a specific source."""
    # Configure per-source rate limits to respect API limits
    rate_configs = {
        DocumentSource.INTERCOM_ARTICLE: RateLimitConfig(requests_per_second=1.0, burst_size=5),
        DocumentSource.JIRA_TASK: RateLimitConfig(requests_per_second=0.5, burst_size=3),
        DocumentSource.CONFLUENCE_PAGE: RateLimitConfig(requests_per_second=2.0, burst_size=10),
        DocumentSource.CUSTOM: RateLimitConfig(requests_per_second=1.0, burst_size=5),
    }
    
    # Default conservative rate limit
    default_config = RateLimitConfig(requests_per_second=0.5, burst_size=3)
    return rate_configs.get(source, default_config)

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
        except Exception:
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

async def _fetch_data_from_source(fetcher: DocumentFetcher, source: DocumentSource, credentials: Credentials, metadata: FetchMetadata, rate_limit_config: RateLimitConfig = None) -> PaginatedDocuments:
    result = await fetcher.fetch_from_source(source, credentials, metadata, rate_limit_config)
    return result

async def _save_data_to_db(fetch_result: PaginatedDocuments):    
    # Store documents in database
    for doc_data in fetch_result.documents:
        try:
            doc_id = await create_document(
                title=doc_data.title,
                content=doc_data.content,
                source=doc_data.source,
                original_id=doc_data.original_id,
                content_url=doc_data.content_url,
                language=doc_data.language,
                metadata=doc_data.metadata
            )
            logger.document(f"Stored document: {doc_data.title} (ID: {doc_id})")
        except Exception as e:
            logger.error(f"❌ Failed to store document {doc_data.title}: {e}")

async def _send_next_batch_message(memory_ingest_dto: MemoryIngestDto, next_metadata: FetchMetadata):
    """Send message for next batch processing."""
    try:
        nc = await nats.connect("nats://localhost:4222")
        js = nc.jetstream()
        
        # Log current pagination metadata for debugging
        logger.info(f"Sending next batch with metadata: {next_metadata.metadata}")
        
        # Create next batch message with updated metadata
        next_batch = MemoryIngestDto(
            source=memory_ingest_dto.source,
            credentials=memory_ingest_dto.credentials,
            metadata=next_metadata
        )
        
        # Publish to the same subject for continuation
        subject = f"siestai.v1.ingest.database.{memory_ingest_dto.source.value}"
        message_json = next_batch.model_dump_json()
        logger.debug(f"Publishing message: {message_json}")
        await js.publish(subject, message_json.encode())
        
        logger.info(f"Queued next batch for {memory_ingest_dto.source.value} with updated metadata")
        await nc.close()
        
    except Exception as e:
        logger.error(f"Failed to send next batch message: {e}")
        raise

async def ingest_to_database(memory_ingest_dto: MemoryIngestDto):
    """Ingest a batch of data to the database."""
    logger.database(f"Starting database ingestion for source: {memory_ingest_dto.source}")
    logger.info(f"Received metadata: {memory_ingest_dto.metadata.metadata}")
    logger.debug(f"Ingestion details: source={memory_ingest_dto.source}, metadata={memory_ingest_dto.metadata}")
    
    fetcher = DocumentFetcher()
    fetcher.register_connector(memory_ingest_dto.source)
    
    try:
        # Configure rate limiting based on source
        rate_limit_config = _get_rate_limit_config(memory_ingest_dto.source)
        
        # Fetch data with rate limiting
        result = await _fetch_data_from_source(
            fetcher, 
            memory_ingest_dto.source, 
            memory_ingest_dto.credentials, 
            memory_ingest_dto.metadata,
            rate_limit_config
        )
        
        logger.document(f"Fetched {len(result.documents)} documents")
        logger.info(f"Result has_more: {result.has_more}")
        logger.info(f"Result metadata after fetch: {result.fetch_metadata.metadata}")
        
        # Save current batch to database
        await _save_data_to_db(result)
        logger.document(f"Saved {len(result.documents)} documents to the database")
        
        # Check if there are more pages to process
        if result.has_more:
            logger.info(f"More data available for {memory_ingest_dto.source.value}, queuing next batch")
            await _send_next_batch_message(memory_ingest_dto, result.fetch_metadata)
        else:
            logger.info(f"All data processed for {memory_ingest_dto.source.value}, ingestion complete")
            
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
