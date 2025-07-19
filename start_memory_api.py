#!/usr/bin/env python3
"""
Startup script for Siestai Memory API
Run with: python start_memory_api.py
Or with poetry: poetry run start-memory-api
"""

import uvicorn
import os
from pathlib import Path

def main():
    """Start the Memory API server."""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Set environment variables if not already set
    os.environ.setdefault("INGEST_STREAM_NAME", "SIESTAI-V1-INGEST")
    os.environ.setdefault("INGEST_STREAM_REPLICAS", "1")
    os.environ.setdefault("INGEST_DB_SUBJECT", "siestai.v1.ingest.database.*")
    os.environ.setdefault("INGEST_KG_SUBJECT", "siestai.v1.ingest.knowledgegraph.*")
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info")
    
    print(f"🚀 Starting Siestai Memory API...")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print(f"📝 Log Level: {log_level}")
    print(f"📁 Project Root: {project_root}")
    print("-" * 50)
    
    # Start the server
    uvicorn.run(
        "app.api.memory_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )

if __name__ == "__main__":
    main() 