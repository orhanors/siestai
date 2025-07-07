"""
Example usage of the Python NATS client service.
This demonstrates how to create streams and publish events using the converted NATS client.
"""

import asyncio
import logging
from app.services.nats import NatsService, NatsStreamConfig


async def main():
    """Example of using the NATS client"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create NATS service instance
    nats_service = NatsService("nats://localhost:4222")
    
    try:
        # Create or update a stream
        stream_config = NatsStreamConfig(
            name="example-stream",
            subjects=["example.events", "example.notifications"],
            storage="memory",
            max_msgs=1000,
            retention="limits"
        )
        
        await nats_service.create_or_update_stream(stream_config)
        print("Stream created/updated successfully!")
        
        # Publish some events
        await nats_service.publish_event("example.events", {
            "event_type": "user_created",
            "user_id": "12345",
            "timestamp": "2024-01-01T00:00:00Z"
        })
        
        await nats_service.publish_event("example.notifications", {
            "notification_type": "email",
            "recipient": "user@example.com",
            "message": "Welcome to our service!"
        })
        
        print("Events published successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await nats_service.disconnect()


async def context_manager_example():
    """Example using the async context manager"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    async with NatsService("nats://localhost:4222") as nats:
        # Create stream
        stream_config = NatsStreamConfig(
            name="context-stream",
            subjects=["context.events"],
            storage="file"
        )
        
        await nats.create_or_update_stream(stream_config)
        
        # Publish event
        await nats.publish_event("context.events", {
            "message": "This is published using context manager"
        })
        
        print("Context manager example completed!")


if __name__ == "__main__":
    # Run the basic example
    asyncio.run(main())
    
    # Run the context manager example
    asyncio.run(context_manager_example()) 