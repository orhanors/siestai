#!/usr/bin/env python3
"""
Startup script for Siestai Knowledge Graph Ingest Job
Run with: python start_kg_ingest.py
Or with poetry: poetry run kg-ingest
"""

import os
import asyncio
import signal
from pathlib import Path
from app.memory.knowledge_graph.kg_ingest_job import STREAM_NAME, KG_INGEST_SUBJECT, create_jetstream_consumer, process_knowledge_graph_ingest

def main():
    """Start the Knowledge Graph Ingest Job."""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Set environment variables if not already set
    os.environ.setdefault("INGEST_KG_SUBJECT", "siestai.v1.ingest.knowledge_graph.*")
    
    # Configuration
    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    log_level = os.getenv("LOG_LEVEL", "info")
    
    print(f"🚀 Starting Siestai Knowledge Graph Ingest Job...")
    print(f"🔌 NATS URL: {nats_url}")
    print(f"📝 Log Level: {log_level}")
    print(f"📁 Project Root: {project_root}")
    print(f"📨 Subject: {os.getenv('INGEST_KG_SUBJECT')}")
    print(f"🕸️  Neo4j URI: {os.getenv('NEO4J_URI', 'bolt://localhost:7687')}")
    print(f"🤖 LLM Model: {os.getenv('LLM_CHOICE', 'gpt-4.1-mini')}")
    print(f"🔤 Embedding Model: {os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')}")
    print("-" * 50)
    
    print("ℹ️  Starting JetStream consumer with queue group")
    print("🔄 Max retries: 20 attempts before termination")
    print("🗑️ Messages will be removed after acknowledgment (WORK_QUEUE retention)")
    print("🧠 Using LangGraph/Graphiti for knowledge graph operations")
    
    # Create event to keep process alive
    stop_event = asyncio.Event()
    
    def signal_handler():
        print("\n🛑 Received shutdown signal...")
        stop_event.set()
    
    async def run():
        # Create JetStream consumer first
        print("🔧 Creating JetStream consumer...")
        await create_jetstream_consumer()
        
        # Start JetStream message processing
        print("✅ Knowledge Graph Ingest Job started successfully")
        print("🔄 Listening for messages...")
        
        try:
            # Start message processing
            await process_knowledge_graph_ingest()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down Knowledge Graph Ingest Job...")
        finally:
            print("✅ Knowledge Graph Ingest Job shutdown complete")
    
    # Run the application
    asyncio.run(run())

if __name__ == "__main__":
    main()