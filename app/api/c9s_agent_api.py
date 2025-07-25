"""FastAPI service for C9S Agent with interactive chat capabilities."""

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json

from app.agents.c9s_agent.c9s_agent import C9SAgent

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="C9S Agent API",
    description="Interactive chat service for C9S Agent with JIRA MCP and web search",
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


class HumanInputRequest(BaseModel):
    """Request model for human input in human-in-the-loop."""
    session_id: str
    feedback: str


class SessionContextResponse(BaseModel):
    """Response model for session context."""
    query: str
    current_step: str
    web_results: List[Dict[str, Any]]
    jira_results: List[Dict[str, Any]]
    next_action: str


@app.on_event("startup")
async def startup_event():
    """Initialize the C9S agent on startup."""
    global agent
    
    try:
        logger.info("Initializing C9S Agent...")
        
        # Configuration from environment variables
        config = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4"),
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
            "tavily_api_key": os.getenv("TAVILY_API_KEY"),
            "jira_mcp_path": os.getenv("JIRA_MCP_PATH"),
            "enable_human_loop": os.getenv("ENABLE_HUMAN_LOOP", "true").lower() == "true",
            "postgres_connection_string": os.getenv("DATABASE_URL")
        }
        
        agent = C9SAgent(**config)
        
        # Initialize the agent in context manager
        await agent.__aenter__()
        
        logger.info("C9S Agent initialized successfully")
        
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


@app.get("/session/{session_id}/context", response_model=SessionContextResponse)
async def get_session_context(session_id: str) -> SessionContextResponse:
    """Get context for a session that requires human input."""
    if not agent:
        raise HTTPException(status_code=503, detail="C9S Agent not initialized")
    
    try:
        logger.info(f"Getting context for session: {session_id}")
        
        context = await agent.get_human_input_context(session_id)
        
        if not context:
            raise HTTPException(status_code=404, detail="Session not found or no context available")
        
        return SessionContextResponse(
            query=context.get("query", ""),
            current_step=context.get("current_step", ""),
            web_results=context.get("web_results", []),
            jira_results=context.get("jira_results", []),
            next_action=context.get("next_action", "")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session context: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/session/continue", response_model=ChatResponse)
async def continue_with_feedback(request: HumanInputRequest) -> ChatResponse:
    """Continue a session with human feedback."""
    if not agent:
        raise HTTPException(status_code=503, detail="C9S Agent not initialized")
    
    try:
        logger.info(f"Continuing session {request.session_id} with human feedback")
        
        result = await agent.continue_with_human_feedback(
            session_id=request.session_id,
            human_feedback=request.feedback
        )
        
        return ChatResponse(
            answer=result["answer"],
            query="", # Not available in continuation
            session_id=result["session_id"],
            requires_human_input=False,
            timestamp=datetime.now(),
            metadata=result["metadata"],
            sources={}  # Not available in continuation
        )
        
    except Exception as e:
        logger.error(f"Error continuing session with feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not agent:
        raise HTTPException(status_code=503, detail="C9S Agent not initialized")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "agent_initialized": agent is not None
    }


@app.get("/status")
async def get_status():
    """Get service status and configuration."""
    config_info = {
        "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
        "jira_mcp_configured": bool(os.getenv("JIRA_MCP_PATH")),
        "database_configured": bool(os.getenv("DATABASE_URL")),
        "human_loop_enabled": os.getenv("ENABLE_HUMAN_LOOP", "true").lower() == "true",
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4")
    }
    
    return {
        "status": "operational",
        "timestamp": datetime.now(),
        "agent_initialized": agent is not None,
        "configuration": config_info
    }


# Streaming chat endpoint for real-time interactions
@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """Stream chat responses for real-time interaction."""
    if not agent:
        raise HTTPException(status_code=503, detail="C9S Agent not initialized")
    
    async def generate_stream():
        """Generate streaming response."""
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing query...'})}\n\n"
            
            # Process the query
            result = await agent.process_query(
                query=request.query,
                user_id=request.user_id,
                profile_id=request.profile_id,
                session_id=request.session_id,
                human_feedback=request.human_feedback
            )
            
            # Send progress updates based on metadata
            if result["metadata"].get("step_results"):
                for step, step_result in result["metadata"]["step_results"].items():
                    yield f"data: {json.dumps({'type': 'step', 'step': step, 'result': step_result})}\n\n"
            
            # Send final response
            response_data = {
                "type": "final",
                "answer": result["answer"],
                "session_id": result["session_id"],
                "requires_human_input": result["requires_human_input"],
                "metadata": result["metadata"],
                "sources": result["sources"]
            }
            
            yield f"data: {json.dumps(response_data)}\n\n"
            
        except Exception as e:
            error_data = {
                "type": "error",
                "message": f"Error processing request: {str(e)}"
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8002"))
    log_level = os.getenv("LOG_LEVEL", "info")
    
    logger.info(f"Starting C9S Agent API on {host}:{port}")
    
    uvicorn.run(
        "app.api.c9s_agent_api:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=os.getenv("ENVIRONMENT") == "development"
    )