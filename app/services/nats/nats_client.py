import asyncio
import json
import logging
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass

import nats
from nats.aio.client import Client as NATSClient
from nats.js import JetStreamContext
from nats.js.api import StreamConfig, StreamInfo


@dataclass
class NatsStreamConfig:
    """Configuration for NATS JetStream stream"""
    name: str
    subjects: list[str]
    storage: str = "file"  # "file" or "memory"
    max_msgs: int = -1
    max_bytes: int = -1
    max_age: int = 0
    retention: str = "limits"  # "limits", "interest", "workqueue"
    discard: str = "old"  # "old" or "new"
    duplicate_window: int = 0
    
    def to_stream_config(self) -> StreamConfig:
        """Convert to NATS StreamConfig"""
        return StreamConfig(
            name=self.name,
            subjects=self.subjects,
            storage=self.storage,
            max_msgs=self.max_msgs,
            max_bytes=self.max_bytes,
            max_age=self.max_age,
            retention=self.retention,
            discard=self.discard,
            duplicate_window=self.duplicate_window,
        )


class NatsService:
    """Python NATS client service with JetStream support"""
    
    def __init__(self, nats_url: str = "nats://localhost:4222"):
        self.nats_url = nats_url
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client: Optional[NATSClient] = None
        self._js: Optional[JetStreamContext] = None
    
    async def connect(self) -> NATSClient:
        """Connect to NATS server if not already connected"""
        if self._client is None or self._client.is_closed:
            try:
                self._client = await nats.connect(self.nats_url)
                self._js = self._client.jetstream()
                self.logger.info(f"Connected to NATS at {self.nats_url}")
            except Exception as e:
                self.logger.error(f"Failed to connect to NATS: {e}")
                raise
        return self._client
    
    async def disconnect(self):
        """Disconnect from NATS server"""
        if self._client and not self._client.is_closed:
            await self._client.close()
            self._client = None
            self._js = None
            self.logger.info("Disconnected from NATS")
    
    async def create_or_update_stream(self, options: NatsStreamConfig):
        """Create or update a NATS JetStream stream"""
        await self.connect()
        
        try:
            stream = await self._get_stream(options)
            stream_config = options.to_stream_config()
            
            if stream:
                # Update existing stream
                await self._js.update_stream(stream_config)
                self.logger.debug(f"Stream {options.name} updated")
            else:
                # Create new stream
                await self._js.add_stream(stream_config)
                self.logger.debug(f"Stream {options.name} created")
                
        except Exception as e:
            self.logger.error(f"Failed to create/update stream {options.name}: {e}")
            raise
    
    async def _get_stream(self, options: NatsStreamConfig) -> Optional[StreamInfo]:
        """Get stream information by name (private method)"""
        try:
            return await self._js.stream_info(options.name)
        except Exception:
            # Stream doesn't exist
            return None
    
    async def publish_event(self, subject: str, data: Any):
        """Publish an event to a subject"""
        await self.connect()
        
        try:
            # Convert data to JSON if it's not already a string or bytes
            if isinstance(data, (str, bytes)):
                payload = data if isinstance(data, bytes) else data.encode()
            else:
                payload = json.dumps(data).encode()
            
            # Publish to JetStream
            await self._js.publish(subject, payload)
            self.logger.debug(f"Event published to {subject}")
            
        except Exception as e:
            self.logger.error(f"Failed to publish event to {subject}: {e}")
            raise
    
    async def publish_event_sync(self, subject: str, data: Any):
        """Publish an event synchronously (blocking)"""
        await self.connect()
        
        try:
            # Convert data to JSON if it's not already a string or bytes
            if isinstance(data, (str, bytes)):
                payload = data if isinstance(data, bytes) else data.encode()
            else:
                payload = json.dumps(data).encode()
            
            # Publish to regular NATS (not JetStream)
            await self._client.publish(subject, payload)
            self.logger.debug(f"Event published to {subject}")
            
        except Exception as e:
            self.logger.error(f"Failed to publish event to {subject}: {e}")
            raise
    
    async def subscribe(self, subject: str, callback, queue: str = None):
        """Subscribe to a subject with a callback"""
        await self.connect()
        
        try:
            sub = await self._client.subscribe(subject, queue=queue, cb=callback)
            self.logger.debug(f"Subscribed to {subject}")
            return sub
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {subject}: {e}")
            raise
    
    async def create_consumer(self, stream_name: str, consumer_name: str, 
                             subjects: list[str] = None):
        """Create a JetStream consumer"""
        await self.connect()
        
        try:
            consumer_config = {
                "name": consumer_name,
                "durable_name": consumer_name,
            }
            
            if subjects:
                consumer_config["filter_subject"] = subjects[0] if len(subjects) == 1 else None
            
            consumer = await self._js.add_consumer(stream_name, **consumer_config)
            self.logger.debug(f"Consumer {consumer_name} created for stream {stream_name}")
            return consumer
        except Exception as e:
            self.logger.error(f"Failed to create consumer {consumer_name}: {e}")
            raise
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
