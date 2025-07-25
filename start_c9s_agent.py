#!/usr/bin/env python3
"""Startup script for C9S Agent service with Claude LLM."""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up environment
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, Optional, List
import asyncio
import json
import uvicorn

# Import the C9S agent
sys.path.insert(0, str(project_root / "app" / "agents" / "c9s-agent"))
from c9s_agent import C9SAgent

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="C9S Agent API with Claude",
    description="C9S Agent with Claude LLM, JIRA MCP, and web search",
    version="1.0.0"
)

# Global agent instance
agent: Optional[C9SAgent] = None


class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    query: str
    user_id: str
    profile_id: str
    session_id: Optional[str] = None
    human_feedback: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    answer: str
    query: str
    session_id: str
    requires_human_input: bool
    timestamp: datetime
    metadata: Dict[str, Any]
    sources: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Initialize the C9S agent on startup."""
    global agent
    
    try:
        logger.info("Initializing C9S Agent with Claude...")
        
        # Configuration from environment variables
        config = {
            "model": os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
            "temperature": float(os.getenv("CLAUDE_TEMPERATURE", "0.1")),
            "tavily_api_key": os.getenv("TAVILY_API_KEY"),
            "jira_mcp_path": os.getenv("JIRA_MCP_PATH"),
            "enable_human_loop": os.getenv("ENABLE_HUMAN_LOOP", "true").lower() == "true",
            "postgres_connection_string": os.getenv("DATABASE_URL")
        }
        
        agent = C9SAgent(**config)
        
        # Initialize the agent in context manager
        await agent.__aenter__()
        
        logger.info("C9S Agent with Claude initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize C9S Agent: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up the C9S agent on shutdown."""
    global agent
    
    if agent:
        try:
            await agent.__aexit__(None, None, None)
            logger.info("C9S Agent shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down C9S Agent: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message through the C9S agent."""
    if not agent:
        raise HTTPException(status_code=503, detail="C9S Agent not initialized")
    
    try:
        logger.info(f"Processing chat request from user {request.user_id}")
        
        # Process the query
        result = await agent.process_query(
            query=request.query,
            user_id=request.user_id,
            profile_id=request.profile_id,
            session_id=request.session_id,
            human_feedback=request.human_feedback
        )
        
        return ChatResponse(
            answer=result["answer"],
            query=result["query"],
            session_id=result["session_id"],
            requires_human_input=result["requires_human_input"],
            timestamp=datetime.now(),
            metadata=result["metadata"],
            sources=result["sources"]
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not agent:
        raise HTTPException(status_code=503, detail="C9S Agent not initialized")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "agent_initialized": agent is not None,
        "version": "claude"
    }


@app.get("/status")
async def get_status():
    """Get service status and configuration."""
    config_info = {
        "anthropic_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
        "jira_mcp_configured": bool(os.getenv("JIRA_MCP_PATH")),
        "database_configured": bool(os.getenv("DATABASE_URL")),
        "human_loop_enabled": os.getenv("ENABLE_HUMAN_LOOP", "true").lower() == "true",
        "claude_model": os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
        "version": "claude"
    }
    
    return {
        "status": "operational",
        "timestamp": datetime.now(),
        "agent_initialized": agent is not None,
        "configuration": config_info
    }


def main():
    """Main entry point for the C9S Agent service."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting C9S Agent API with Claude...")
    
    # Check required environment variables
    required_env_vars = {
        "ANTHROPIC_API_KEY": "Anthropic API key for Claude LLM",
    }
    
    optional_env_vars = {
        "TAVILY_API_KEY": "Tavily API key for web search",
        "JIRA_MCP_PATH": "Path to JIRA MCP server executable",
        "DATABASE_URL": "PostgreSQL connection string for memory",
        "ENABLE_HUMAN_LOOP": "Enable human-in-the-loop (default: true)",
        "CLAUDE_MODEL": "Claude model to use (default: claude-3-5-sonnet-20241022)",
        "CLAUDE_TEMPERATURE": "Model temperature (default: 0.1)"
    }
    
    # Check required environment variables
    missing_required = []
    for var, description in required_env_vars.items():
        if not os.getenv(var):
            missing_required.append(f"  {var}: {description}")
    
    if missing_required:
        logger.error("Missing required environment variables:")
        for var in missing_required:
            logger.error(var)
        sys.exit(1)
    
    # Log optional environment variables status
    logger.info("Environment configuration:")
    for var, description in optional_env_vars.items():
        value = os.getenv(var)
        status = "✓ Configured" if value else "✗ Not configured"
        logger.info(f"  {var}: {status} - {description}")
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8002"))
    log_level = os.getenv("LOG_LEVEL", "info")
    
    logger.info(f"Starting C9S Agent API server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        reload=os.getenv("ENVIRONMENT") == "development"
    )


if __name__ == "__main__":
    main()