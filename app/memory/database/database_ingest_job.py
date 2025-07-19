from faststream.nats import NatsBroker
from faststream import Logger
from faststream.nats import NatsMessage
from pydantic import BaseModel, Field
import os
import asyncio

class Task(BaseModel):
    id: int = Field(..., example=1)
    payload: str = Field(..., example="do something")

broker = NatsBroker("nats://localhost:4222")

DB_INGEST_SUBJECT = os.getenv("INGEST_DB_SUBJECT", "siestai.v1.ingest.database.*")
STREAM_NAME = os.getenv("INGEST_STREAM_NAME", "SIESTAI-V1-INGEST")

@broker.subscriber(
    DB_INGEST_SUBJECT, 
    description="Handle database ingest tasks"
)
async def handle_task(msg: Task, logger: Logger, raw_msg: NatsMessage):
    # Get delivery count from headers
    delivery_count = raw_msg.headers.get("Nats-Msg-Delivery-Count", 0) if raw_msg.headers else 0
    max_retries = 20
    
    logger.info(f"Processing task {msg.id}: {msg.payload} (attempt {delivery_count + 1}/{max_retries})")
    
    try:
        # Simulate some work
        await asyncio.sleep(0.1)  # Simulate processing time
        
        logger.info(f"✅ Task {msg.id} completed successfully")
        
        # Explicitly acknowledge the message
        await raw_msg.ack()
        logger.info(f"📨 Message {msg.id} acknowledged")
        
        # Show stream status after processing
        try:
            import subprocess
            result = subprocess.run(['nats', 'stream', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"📊 Stream Status after processing {msg.id}:")
                for line in result.stdout.strip().split('\n'):
                    if 'SIESTAI-V1-INGEST' in line:
                        logger.info(f"   {line.strip()}")
            else:
                logger.warning("Could not get stream status")
        except Exception as e:
            logger.warning(f"Could not get stream status: {e}")
        
    except Exception as e:
        logger.error(f"❌ Error processing task {msg.id}: {e}")
        
        # Check if we've exceeded max retries
        if delivery_count >= max_retries:
            logger.error(f"🚨 Task {msg.id} exceeded max retries ({max_retries}). Terminating message.")
            await raw_msg.ack()  # Acknowledge to prevent infinite redelivery
            logger.error(f"📨 Message {msg.id} terminated after {max_retries} attempts")
        else:
            # Negative acknowledgment for failed messages (will be redelivered)
            await raw_msg.nak()
            logger.warning(f"📨 Message {msg.id} negatively acknowledged (will retry)")
        
        raise