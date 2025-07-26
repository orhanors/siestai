# C9S Agent

A LangGraph-based intelligent agent with JIRA MCP integration, web search capabilities, and human-in-the-loop functionality.

## Features

- **LangGraph Architecture**: Built on LangGraph for stateful, multi-step processing
- **JIRA MCP Integration**: Connect to JIRA via Model Context Protocol (MCP) server
- **Web Search**: Powered by Tavily for current information retrieval
- **Human-in-the-Loop**: Interactive decision making with interrupts
- **PostgreSQL Memory**: Long-term memory storage using LangGraph checkpointer
- **Router Component**: Intelligent routing to appropriate tools based on query analysis

## Architecture

### Core Components

1. **Router**: Analyzes queries and routes to appropriate tools
2. **Web Search**: Tavily-powered web search for current information
3. **JIRA Search**: MCP-based JIRA integration for ticket management
4. **Human Check**: Determines when human input is needed
5. **Synthesizer**: Combines information from all sources

### State Management

The agent maintains state through LangGraph's PostgreSQL checkpointer, enabling:
- Conversation continuity across sessions
- Recovery from interruptions
- Long-term memory storage

## Configuration

### Environment Variables

**Required:**
- `OPENAI_API_KEY`: OpenAI API key for LLM
- `DATABASE_URL`: PostgreSQL connection string

**Optional:**
- `TAVILY_API_KEY`: Tavily API key for web search
- `JIRA_MCP_PATH`: Path to JIRA MCP server executable
- `ENABLE_HUMAN_LOOP`: Enable human-in-the-loop (default: true)
- `OPENAI_MODEL`: OpenAI model (default: gpt-4)
- `OPENAI_TEMPERATURE`: Model temperature (default: 0.1)

### JIRA MCP Setup

1. Install the JIRA MCP server from: https://github.com/sooperset/mcp-atlassian
2. Set `JIRA_MCP_PATH` to the server executable path
3. Configure JIRA credentials as required by the MCP server

## Usage

### Starting the Service

```bash
# Using Poetry script
poetry run c9s-agent

# Or directly
python start_c9s_agent.py
```

### API Endpoints

- `POST /chat`: Process a chat message
- `GET /session/{session_id}/context`: Get session context for human input
- `POST /session/continue`: Continue with human feedback
- `POST /chat/stream`: Streaming chat responses
- `GET /health`: Health check
- `GET /status`: Service status and configuration

### Example Usage

```python
from app.agents.c9s_agent import C9SAgent

async with C9SAgent() as agent:
    result = await agent.process_query(
        query="Find JIRA tickets related to bug fixes",
        user_id="user123",
        profile_id="profile456"
    )
    print(result["answer"])
```

## Human-in-the-Loop

The agent automatically requests human input for:
- Sensitive operations (delete, close, assign)
- Queries with no results
- Ambiguous requests

When human input is required:
1. The agent pauses execution
2. Context is provided via `/session/{session_id}/context`
3. Human provides feedback via `/session/continue`
4. Execution resumes with the feedback

## Integration with Siestai

This agent integrates with the existing Siestai infrastructure:
- Uses the same PostgreSQL database
- Follows Bootstrap lifecycle management
- Integrates with the embedding service
- Compatible with the existing API patterns

## Development

### Testing

```bash
# Install dependencies
poetry install

# Run tests (when available)
poetry run pytest app/agents/c9s-agent/tests/

# Start in development mode
ENVIRONMENT=development poetry run c9s-agent
```

### Adding New Tools

1. Create a new method in `C9SAgent`
2. Add it as a node in `_build_graph()`
3. Update routing logic in `route_query()`
4. Add appropriate edges in the graph

## Troubleshooting

### Common Issues

1. **JIRA MCP Connection Failed**
   - Verify `JIRA_MCP_PATH` points to correct executable
   - Check JIRA MCP server logs
   - Ensure JIRA credentials are configured

2. **Database Connection Issues**
   - Verify `DATABASE_URL` is correct
   - Ensure PostgreSQL is running
   - Check database permissions

3. **Tavily API Errors**
   - Verify `TAVILY_API_KEY` is set
   - Check API quota limits
   - Review Tavily API documentation

### Logging

The agent uses structured logging with the following levels:
- INFO: General operation information
- WARNING: Non-critical issues
- ERROR: Critical errors that affect functionality

Logs include step-by-step execution tracking for debugging.