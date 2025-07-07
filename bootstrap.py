"""
Application Bootstrap System
Handles application startup, shutdown, and lifecycle management.
"""

import asyncio
import logging
from typing import List, Callable, Awaitable
from dataclasses import dataclass
from contextlib import asynccontextmanager
import os
from app.memory.database.database import initialize_database, close_database
from app.services.nats import NatsService
from app.services.nats.stream_configs import NATS_STREAM_CONFIGS
from app.utils.logger import setup_logging

# Type definitions
StartupFunc = Callable[[], Awaitable[None]]
ShutdownFunc = Callable[[], Awaitable[None]]


@dataclass
class AppConfig:
    """Application configuration"""
    debug: bool = False
    nats_url: str = "nats://localhost:4222"
    database_url: str = "postgresql://localhost:5432/siestai"
    log_level: str = "INFO"


class ApplicationBootstrap:
    """Application lifecycle management"""
    
    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()
        self.logger = logging.getLogger(__name__)
        self.startup_hooks: List[StartupFunc] = []
        self.shutdown_hooks: List[ShutdownFunc] = []
        self._nats_service: NatsService = None
        
    def add_startup_hook(self, func: StartupFunc):
        """Add a function to run on startup"""
        self.startup_hooks.append(func)
        
    def add_shutdown_hook(self, func: ShutdownFunc):
        """Add a function to run on shutdown"""
        self.shutdown_hooks.append(func)
        
    async def initialize_logging(self):
        """Initialize application logging"""
        setup_logging(level=self.config.log_level)
        self.logger.info("Logging initialized")
        
    async def initialize_database(self):
        """Initialize database connections"""
        await initialize_database()
        self.logger.info("Database initialized")
        
    async def initialize_nats(self):
        """Initialize NATS service and streams"""
        self._nats_service = NatsService(self.config.nats_url)
        
        # Create default streams
        stream_configs = NATS_STREAM_CONFIGS
        
        for stream_config in stream_configs:
            await self._nats_service.create_or_update_stream(stream_config)
            
        self.logger.info("NATS service initialized")
        
    async def cleanup_database(self):
        """Clean up database connections"""
        await close_database()
        self.logger.info("Database connections closed")
        
    async def cleanup_nats(self):
        """Clean up NATS connections"""
        if self._nats_service:
            await self._nats_service.disconnect()
            self.logger.info("NATS service disconnected")
            
    async def startup(self):
        """Run all startup procedures"""
        self.logger.info("Starting application bootstrap...")
        
        # Core services
        await self.initialize_logging()
        await self.initialize_database()
        await self.initialize_nats()
        
        # Custom startup hooks
        for hook in self.startup_hooks:
            try:
                await hook()
                self.logger.info(f"Startup hook {hook.__name__} completed")
            except Exception as e:
                self.logger.error(f"Startup hook {hook.__name__} failed: {e}")
                raise
                
        self.logger.info("Application bootstrap completed successfully")
        
    async def shutdown(self):
        """Run all shutdown procedures"""
        self.logger.info("Starting application shutdown...")
        
        # Custom shutdown hooks (reverse order)
        for hook in reversed(self.shutdown_hooks):
            try:
                await hook()
                self.logger.info(f"Shutdown hook {hook.__name__} completed")
            except Exception as e:
                self.logger.error(f"Shutdown hook {hook.__name__} failed: {e}")
                
        # Core services cleanup
        await self.cleanup_nats()
        await self.cleanup_database()
        
        self.logger.info("Application shutdown completed")
        
    @asynccontextmanager
    async def lifespan(self):
        """Context manager for application lifecycle"""
        try:
            await self.startup()
            yield self
        finally:
            await self.shutdown()
            
    def get_nats_service(self) -> NatsService:
        """Get the NATS service instance"""
        return self._nats_service


# Global application instance
app_bootstrap = ApplicationBootstrap()


# Decorator for registering startup hooks
def startup_hook(func: StartupFunc):
    """Decorator to register a startup hook"""
    app_bootstrap.add_startup_hook(func)
    return func


# Decorator for registering shutdown hooks  
def shutdown_hook(func: ShutdownFunc):
    """Decorator to register a shutdown hook"""
    app_bootstrap.add_shutdown_hook(func)
    return func 