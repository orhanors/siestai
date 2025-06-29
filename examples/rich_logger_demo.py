#!/usr/bin/env python3
"""
Demo script for the Rich logger functionality.
"""

import asyncio
import time
from pathlib import Path

# Add the app directory to the path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.utils.logger import (
    SiestaiLogger, 
    ProgressLogger, 
    StatusLogger, 
    setup_logging,
    logger,
    db_logger,
    api_logger,
    intercom_logger
)
from app.config.logging_config import get_logging_config, get_color_theme


def demo_basic_logging():
    """Demonstrate basic logging functionality."""
    print("\n" + "="*60)
    print("BASIC LOGGING DEMO")
    print("="*60)
    
    # Setup logging
    setup_logging(level="DEBUG")
    
    # Show startup banner
    StatusLogger.show_startup_banner()
    
    # Basic log messages
    logger.info("Starting basic logging demo")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.success("This is a success message!")
    
    # Component-specific logging
    db_logger.database("Connected to PostgreSQL database")
    api_logger.api("Making API request to external service")
    intercom_logger.intercom("Fetching articles from Intercom")
    
    logger.info("Basic logging demo completed")


def demo_status_display():
    """Demonstrate status display functionality."""
    print("\n" + "="*60)
    print("STATUS DISPLAY DEMO")
    print("="*60)
    
    # System status table
    status_data = {
        "Database": {
            "status": "ok", 
            "details": "PostgreSQL 15.2 - Connected"
        },
        "Intercom API": {
            "status": "ok", 
            "details": "Rate limit: 85/100 requests"
        },
        "Vector Search": {
            "status": "ok", 
            "details": "pgvector extension active"
        },
        "File Storage": {
            "status": "error", 
            "details": "Permission denied on /data"
        },
        "Redis Cache": {
            "status": "ok", 
            "details": "Connected to localhost:6379"
        }
    }
    
    StatusLogger.show_status_table(status_data)
    
    # Document statistics
    stats = {
        "total_documents": 1250,
        "documents_with_embeddings": 1180,
        "unique_sources": {
            "intercom": 450, 
            "jira": 300, 
            "confluence": 500
        },
        "languages": {
            "en": 1000, 
            "es": 150, 
            "fr": 100
        },
        "avg_document_size": "2.3KB",
        "last_updated": "2024-01-15 14:30:00"
    }
    
    StatusLogger.show_document_stats(stats)


def demo_progress_tracking():
    """Demonstrate progress tracking functionality."""
    print("\n" + "="*60)
    print("PROGRESS TRACKING DEMO")
    print("="*60)
    
    # Single task progress
    with ProgressLogger("Processing documents") as progress:
        task = progress.add_task("Fetching from Intercom", total=100)
        
        for i in range(10):
            progress.update(task, advance=10)
            time.sleep(0.2)
        
        progress.complete(task)
    
    # Multiple tasks progress
    with ProgressLogger("Multi-step processing") as progress:
        # Task 1: Fetch data
        fetch_task = progress.add_task("Fetching data", total=50)
        for i in range(5):
            progress.update(fetch_task, advance=10)
            time.sleep(0.1)
        
        # Task 2: Process data
        process_task = progress.add_task("Processing data", total=30)
        for i in range(3):
            progress.update(process_task, advance=10)
            time.sleep(0.1)
        
        # Task 3: Save results
        save_task = progress.add_task("Saving results", total=20)
        for i in range(2):
            progress.update(save_task, advance=10)
            time.sleep(0.1)
        
        # Complete all tasks
        progress.complete(fetch_task)
        progress.complete(process_task)
        progress.complete(save_task)


async def demo_async_logging():
    """Demonstrate async logging functionality."""
    print("\n" + "="*60)
    print("ASYNC LOGGING DEMO")
    print("="*60)
    
    async def simulate_api_call(name: str, duration: float):
        """Simulate an API call."""
        logger.info(f"Starting {name} API call")
        await asyncio.sleep(duration)
        logger.success(f"Completed {name} API call")
        return f"Result from {name}"
    
    async def simulate_database_operation(name: str, duration: float):
        """Simulate a database operation."""
        db_logger.database(f"Starting {name} database operation")
        await asyncio.sleep(duration)
        db_logger.database(f"Completed {name} database operation")
        return f"DB result from {name}"
    
    # Run multiple async operations
    tasks = [
        simulate_api_call("Intercom", 1.0),
        simulate_api_call("Jira", 0.8),
        simulate_database_operation("Document Insert", 0.5),
        simulate_database_operation("Vector Search", 0.3)
    ]
    
    results = await asyncio.gather(*tasks)
    logger.info(f"All async operations completed: {results}")


def demo_error_handling():
    """Demonstrate error handling and tracebacks."""
    print("\n" + "="*60)
    print("ERROR HANDLING DEMO")
    print("="*60)
    
    def function_that_raises_error():
        """Function that raises an error to demonstrate rich tracebacks."""
        x = 10
        y = 0
        return x / y  # This will raise a ZeroDivisionError
    
    try:
        function_that_raises_error()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        # Rich will automatically format the traceback nicely


def demo_configuration():
    """Demonstrate logging configuration."""
    print("\n" + "="*60)
    print("CONFIGURATION DEMO")
    print("="*60)
    
    # Show different environment configurations
    environments = ["development", "staging", "production"]
    
    for env in environments:
        config = get_logging_config(env)
        color_theme = get_color_theme(env)
        
        print(f"\nEnvironment: {env.upper()}")
        print(f"Log Level: {config['level']}")
        print(f"Console Enabled: {config['console_enabled']}")
        print(f"File Enabled: {config['file_enabled']}")
        print(f"Color Theme: {list(color_theme.keys())}")


def main():
    """Run all demos."""
    print("🚀 Rich Logger Demo for Siestai")
    print("="*60)
    
    # Run all demos
    demo_basic_logging()
    demo_status_display()
    demo_progress_tracking()
    
    # Run async demo
    asyncio.run(demo_async_logging())
    
    demo_error_handling()
    demo_configuration()
    
    print("\n" + "="*60)
    print("✅ All demos completed!")
    print("="*60)


if __name__ == "__main__":
    main() 