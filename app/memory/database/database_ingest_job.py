from faststream.nats import NatsBroker
from faststream import Logger
from faststream.nats import NatsMessage
from pydantic import BaseModel, Field
import os
import asyncio
import nats
from nats.js.api import ConsumerConfig
import json

class Task(BaseModel):
    id: int = Field(..., example=1)
    payload: str = Field(..., example="do something")

broker = NatsBroker("nats://localhost:4222")

DB_INGEST_SUBJECT = os.getenv("INGEST_DB_SUBJECT", "siestai.v1.ingest.database.*")
STREAM_NAME = os.getenv("INGEST_STREAM_NAME", "SIESTAI-V1-MEMORY-INGEST")
CONSUMER_NAME = os.getenv("INGEST_CONSUMER_NAME", "siestai-database-ingest-job")
QUEUE_GROUP = "db-ingest-group"

async def create_jetstream_consumer():
    """Create JetStream consumer if it doesn't exist."""
    try:
        # Connect to NATS
        nc = await nats.connect("nats://localhost:4222")
        js = nc.jetstream()
        
        # Check if consumer exists
        try:
            await js.consumer_info(STREAM_NAME, CONSUMER_NAME)
            print(f"✅ Consumer '{CONSUMER_NAME}' already exists")
        except:
            # Create consumer with minimal configuration
            consumer_config = ConsumerConfig(
                durable_name=CONSUMER_NAME,
                filter_subject=DB_INGEST_SUBJECT,
                ack_policy="explicit",
                deliver_group=QUEUE_GROUP,
            )
            
            # Create consumer
            await js.add_consumer(STREAM_NAME, consumer_config)
            print(f"✅ Created JetStream consumer '{CONSUMER_NAME}'")
        
        await nc.close()
        
    except Exception as e:
        print(f"❌ Error creating consumer: {e}")
        raise

async def process_jetstream_messages():
    """Process messages from JetStream consumer."""
    try:
        # Connect to NATS
        nc = await nats.connect("nats://localhost:4222")
        js = nc.jetstream()
        
        # Subscribe to the consumer
        subscription = await js.pull_subscribe(
            subject=DB_INGEST_SUBJECT,
            durable=CONSUMER_NAME,
            stream=STREAM_NAME
        )
        
        print(f"🔄 Starting JetStream message processing...")
        
        while True:
            try:
                # Pull messages
                messages = await subscription.fetch(batch=1, timeout=1)
                
                for msg in messages:
                    try:
                        # Parse the message
                        data = json.loads(msg.data.decode())
                        task = Task(**data)
                        
                        # Get delivery count from headers
                        delivery_count = msg.header.get("Nats-Msg-Delivery-Count", 0) if msg.header else 0
                        max_retries = 20
                        
                        print(f"Processing task {task.id}: {task.payload} (attempt {delivery_count + 1}/{max_retries})")
                        
                        # Simulate some work
                        await asyncio.sleep(0.1)
                        
                        print(f"✅ Task {task.id} completed successfully")
                        
                        # Acknowledge the message
                        await msg.ack()
                        print(f"📨 Message {task.id} acknowledged")
                        
                    except Exception as e:
                        print(f"❌ Error processing message: {e}")
                        # Negative acknowledgment for failed messages
                        await msg.nak()
                        print(f"📨 Message negatively acknowledged (will retry)")
                
            except Exception as e:
                # No messages available, continue
                pass
                
    except Exception as e:
        print(f"❌ Error in JetStream processing: {e}")
        raise
