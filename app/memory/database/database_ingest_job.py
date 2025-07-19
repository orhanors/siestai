from faststream.nats import NatsBroker
from faststream import Logger
from pydantic import BaseModel, Field
import os

class Task(BaseModel):
    id: int = Field(..., example=1)
    payload: str = Field(..., example="do something")

broker = NatsBroker("nats://localhost:4222")

DB_INGEST_SUBJECT = os.getenv("INGEST_DB_SUBJECT", "siestai.v1.ingest.database.*")

@broker.subscriber(DB_INGEST_SUBJECT, description="Handle database ingest tasks")
async def handle_task(msg: Task, logger: Logger):
    logger.info(f"Processing task {msg.id}: {msg.payload}")
    
    # Simulate some work
    import time
    time.sleep(0.1)  # Simulate processing time
    
    logger.info(f"✅ Task {msg.id} completed successfully")
    return None  # This will auto-acknowledge the message