"""
HTTP API app with NATS JetStream
Run with:
    uvicorn src.api_app:app --reload --host 0.0.0.0 --port 8001
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from faststream.nats import NatsBroker
from faststream.nats import RetentionPolicy, StorageType
import asyncio
from contextlib import asynccontextmanager
import os
from typing import Dict, Any, Optional
from app.utils.logger import api_logger, logger
from app.types.document_types import DocumentSource, Credentials
from app.memory.database.database_ingest_job import Task

class MemoryIngestDto(BaseModel):
    source: DocumentSource = Field(..., description="The source to fetch all the data from")
    credentials: Credentials = Field(..., description="The credentials to fetch the data from the source")

class StreamInfo(BaseModel):
    name: str
    subjects: list[str]
    messages: int
    bytes: int
    consumer_count: int

STREAM_NAME = os.getenv("INGEST_STREAM_NAME", "SIESTAI-V1-MEMORY-INGEST")
STREAM_REPLICAS = int(os.getenv("INGEST_STREAM_REPLICAS", "1"))  # Fixed: default to 1 for non-clustered mode
DB_INGEST_SUBJECT = os.getenv("INGEST_DB_SUBJECT", "siestai.v1.ingest.database.*")
KG_INGEST_SUBJECT = os.getenv("INGEST_KG_SUBJECT", "siestai.v1.ingest.knowledgegraph.*")

broker = NatsBroker("nats://localhost:4222")
to_tasks = broker.publisher("test.v1.task", description="Publish task to JetStream")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Siestai Memory API...")
    api_logger.info("Connecting to NATS broker...")
    await broker.connect()
    api_logger.success("Connected to NATS broker successfully")
    
    # Initialize JetStream stream using the broker's JetStream client
    try:
        js = broker._connection.jetstream()
        
        # First, try to get stream info to check if it exists
        try:
            stream_info = await js.stream_info(STREAM_NAME)
            api_logger.success(f"JetStream stream '{STREAM_NAME}' already exists")
            logger.info(f"Stream details: {stream_info.config.subjects}")
        except Exception:
            # Stream doesn't exist, create it
            api_logger.info(f"Creating JetStream stream '{STREAM_NAME}'...")
            
            # Create stream configuration with memory storage
            stream_config = {
                "name": STREAM_NAME,
                "subjects": [DB_INGEST_SUBJECT, KG_INGEST_SUBJECT],
                "description": "Siestai memory data ingestion stream",
                "retention": RetentionPolicy.WORK_QUEUE,  # Remove message after acknowledgment
                "max_msgs": 10000,  # Keep max 10k messages
                "max_age": 3600,  # Keep messages for 1 hour
                "storage": StorageType.FILE,  
                "num_replicas": STREAM_REPLICAS 
            }
            
            await js.add_stream(**stream_config)
            api_logger.success(f"JetStream stream '{STREAM_NAME}' created successfully")
            logger.info(f"Stream configured with subjects: {stream_config['subjects']}")
            
    except Exception as e:
        api_logger.error(f"Error with JetStream stream: {e}")
        logger.warning("Continuing without JetStream - messages will be published to NATS core only")
    
    logger.success("Memory API startup complete")
    yield
    
    # Shutdown
    api_logger.info("Shutting down Memory API...")
    await broker.close()
    logger.info("Memory API shutdown complete")

app = FastAPI(
    title="Siestai Memory API",
    description="API for managing memory and document ingestion with NATS JetStream",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check NATS connection
        if not broker._connection.is_connected():
            raise HTTPException(status_code=503, detail="NATS connection lost")
        
        api_logger.info("Health check requested")
        return {
            "status": "healthy",
            "service": "siestai-memory-api",
            "nats_connected": broker._connection.is_connected(),
            "stream_name": STREAM_NAME
        }
    except Exception as e:
        api_logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/stream/info", response_model=StreamInfo)
async def get_stream_info():
    """Get information about the JetStream stream."""
    try:
        js = broker._connection.jetstream()
        stream_info = await js.stream_info(STREAM_NAME)
        
        api_logger.info("Stream info requested")
        return StreamInfo(
            name=stream_info.config.name,
            subjects=stream_info.config.subjects,
            messages=stream_info.state.messages,
            bytes=stream_info.state.bytes,
            consumer_count=stream_info.state.consumer_count
        )
    except Exception as e:
        api_logger.error(f"Failed to get stream info: {e}")
        raise HTTPException(status_code=404, detail=f"Stream not found: {str(e)}")

@app.post("/publish-task")
async def publish_task_endpoint(task: Task):
    """Publish a task to the NATS stream."""
    api_logger.info(f"Publishing task {task.id} to stream")
    await to_tasks.publish(task)
    api_logger.success(f"Task {task.id} published successfully")
    return {"status": "ok", "id": task.id, "stream": STREAM_NAME}

@app.post("/publish-db-ingest")
async def publish_db_ingest_task(task: MemoryIngestDto):
    """Publish a database ingestion task."""
    try:
        api_logger.info(f"Attempting to publish DB ingest task with source: {task.source}")
        await broker.publish(task, subject=DB_INGEST_SUBJECT)
        api_logger.info(f"Successfully published DB ingest task with source: {task.source}")
        return {"status": "ok", "source": task.source, "subject": DB_INGEST_SUBJECT}
    except Exception as e:
        api_logger.error(f"Failed to publish DB ingest task with source: {task.source}, error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to publish task: {str(e)}")

@app.post("/publish-kg-ingest")
async def publish_kg_ingest_task(task: Task):
    """Publish a knowledge graph ingestion task."""
    api_logger.info(f"Publishing KG ingest task {task.id}")
    await broker.publish(task, subject=KG_INGEST_SUBJECT)
    api_logger.success(f"KG ingest task {task.id} published successfully")
    return {"status": "ok", "id": task.id, "subject": KG_INGEST_SUBJECT}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Siestai Memory API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "stream_info": "/stream/info",
            "publish_task": "/publish-task",
            "publish_db_ingest": "/publish-db-ingest",
            "publish_kg_ingest": "/publish-kg-ingest"
        },
        "stream": STREAM_NAME,
        "subjects": [DB_INGEST_SUBJECT, KG_INGEST_SUBJECT]
    } 