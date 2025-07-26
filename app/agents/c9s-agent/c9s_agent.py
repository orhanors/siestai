"""C9S Agent with LangGraph, JIRA MCP tools, and human-in-the-loop capabilities."""

import os
import logging
import asyncio
import re
from datetime import datetime
from typing import TypedDict, List, Optional, Dict, Any, Callable, Literal
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from .langgraph_compat import StateGraphCompat as StateGraph, START, END, MockPostgresCheckpointer as PostgresCheckpointer
except ImportError:
    from langgraph_compat import StateGraphCompat as StateGraph, START, END, MockPostgresCheckpointer as PostgresCheckpointer
from langchain_anthropic import ChatAnthropic
from tavily import TavilyClient
from langchain.schema import SystemMessage, HumanMessage
# MCP imports
import subprocess
import json
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# Memory and chat history imports
from app.memory.history.session_manager import ChatSession, memory_manager
from app.services.embedding_service import get_embeddings
from app.memory.database.database import vector_search
from app.types.document_types import DocumentSource

logger = logging.getLogger(__name__)


class C9SAgentState(TypedDict):
    """State for the C9S agent."""
    query: str
    user_id: Optional[str]
    session_id: str
    profile_id: str
    messages: List[Dict[str, Any]]
    
    # Tool results
    web_results: List[Dict[str, Any]]
    jira_results: List[Dict[str, Any]]
    document_results: List[Dict[str, Any]]
    
    # Router decision
    next_action: Optional[str]
    tools_to_call: Optional[List[str]]
    
    # Human-in-the-loop
    requires_human_input: bool
    human_feedback: Optional[str]
    
    # Final response
    final_answer: str
    
    # Step tracking
    current_step: str
    step_results: Dict[str, Any]
    step_timings: Dict[str, float]
    
    # Memory context from chat history
    memory_context: Optional[Dict[str, Any]]
    
    # Query refinement mode
    query_refinement_mode: Optional[bool]
    original_query: Optional[str]
    current_jql: Optional[str]


class C9SAgent:
    """LangGraph-based C9S agent with JIRA MCP, web search, and human-in-the-loop."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.1,
        tavily_api_key: Optional[str] = None,
        jira_mcp_path: Optional[str] = None,
        enable_human_loop: bool = True,
        postgres_connection_string: Optional[str] = None
    ):
        """Initialize the C9S agent.
        
        Args:
            model: Claude model to use
            temperature: Model temperature
            tavily_api_key: Tavily API key for web search
            jira_mcp_path: Path to JIRA MCP server
            enable_human_loop: Whether to enable human-in-the-loop
            postgres_connection_string: PostgreSQL connection string for checkpointing
        """
        self.llm = ChatAnthropic(model=model, temperature=temperature)
        self.enable_human_loop = enable_human_loop
        
        # Initialize Tavily client
        self.tavily_client = TavilyClient(api_key=tavily_api_key or os.getenv("TAVILY_API_KEY"))
        
        # JIRA MCP configuration
        self.jira_mcp_path = jira_mcp_path or os.getenv("JIRA_MCP_PATH")
        self.jira_client = None
        
        # PostgreSQL checkpointer for memory
        if postgres_connection_string:
            self.checkpointer = PostgresCheckpointer.from_conn_string(postgres_connection_string)
        else:
            # Use project's existing database connection
            db_url = os.getenv("DATABASE_URL", "postgresql://siestai_user:siestai_password@localhost:5432/siestai_dev")
            self.checkpointer = PostgresCheckpointer.from_conn_string(db_url)
        
        # Chat session management for long-term memory
        self.chat_sessions: Dict[str, ChatSession] = {}
        
        # Cost tracking
        self.total_cost = 0.0
        self.cost_per_token = {
            "claude-3-5-sonnet-20241022": {"input": 0.000003, "output": 0.000015},  # $3/M input, $15/M output
            "claude-3-sonnet-20240229": {"input": 0.000003, "output": 0.000015},
            "claude-3-opus-20240229": {"input": 0.000015, "output": 0.000075},  # $15/M input, $75/M output
            "claude-3-haiku-20240307": {"input": 0.00000025, "output": 0.00000125},  # $0.25/M input, $1.25/M output
        }
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _track_llm_cost(self, response, model: str = None):
        """Track the cost of an LLM call."""
        if not response:
            return
        
        model = model or self.llm.model
        if model not in self.cost_per_token:
            logger.warning(f"Unknown model for cost tracking: {model}")
            return
        
        # Try different ways to get usage information
        usage = None
        input_tokens = 0
        output_tokens = 0
        
        # Method 1: Direct usage attribute
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            input_tokens = getattr(usage, 'prompt_tokens', 0)
            output_tokens = getattr(usage, 'completion_tokens', 0)
        
        # Method 2: Response metadata
        elif hasattr(response, 'response_metadata') and response.response_metadata:
            metadata = response.response_metadata
            if 'usage' in metadata:
                usage_data = metadata['usage']
                input_tokens = usage_data.get('prompt_tokens', 0)
                output_tokens = usage_data.get('completion_tokens', 0)
        
        # Method 3: Direct attributes on response
        elif hasattr(response, 'prompt_tokens') and hasattr(response, 'completion_tokens'):
            input_tokens = response.prompt_tokens
            output_tokens = response.completion_tokens
        
        # Method 4: Check for Anthropic specific structure
        elif hasattr(response, 'content') and hasattr(response, 'usage'):
            usage = response.usage
            input_tokens = getattr(usage, 'input_tokens', 0)
            output_tokens = getattr(usage, 'output_tokens', 0)
        
        # If we still don't have tokens, try to estimate from content length
        if input_tokens == 0 and output_tokens == 0:
            # Rough estimation: 1 token ≈ 4 characters
            if hasattr(response, 'content'):
                content_length = len(response.content)
                output_tokens = max(1, content_length // 4)  # Estimate output tokens
                input_tokens = max(1, output_tokens * 2)  # Estimate input tokens (usually longer)
                logger.info(f"Estimated tokens from content length: input={input_tokens}, output={output_tokens}")
        
        if input_tokens > 0 or output_tokens > 0:
            input_cost = input_tokens * self.cost_per_token[model]["input"]
            output_cost = output_tokens * self.cost_per_token[model]["output"]
            total_call_cost = input_cost + output_cost
            
            self.total_cost += total_call_cost
            
            logger.info(f"LLM call cost: ${total_call_cost:.6f} (input: {input_tokens}, output: {output_tokens})")
            logger.info(f"Total cost so far: ${self.total_cost:.6f}")
        else:
            logger.warning("Could not determine token usage for cost tracking")
    
    def get_total_cost(self) -> float:
        """Get the total cost of all LLM calls in this session."""
        return self.total_cost
    
    def get_cost_summary(self) -> str:
        """Get a formatted cost summary."""
        return f"💰 Total LLM Cost: ${self.total_cost:.6f}"
    
    async def _get_or_create_chat_session(
        self,
        user_id: str,
        profile_id: str,
        session_id: Optional[str] = None
    ) -> ChatSession:
        """Get or create a chat session for long-term memory."""
        session_key = f"{user_id}_{profile_id}_{session_id or 'default'}"
        
        if session_key not in self.chat_sessions:
            # Initialize memory manager if not already done
            if not hasattr(memory_manager, '_initialized'):
                await memory_manager.initialize()
                memory_manager._initialized = True
            
            # Get or create chat session
            chat_session = await memory_manager.get_or_create_session(
                user_id=user_id,
                profile_id=profile_id,
                session_id=session_id,
                session_name=f"C9S Agent Session - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                max_context_messages=20,
                similarity_threshold=0.8
            )
            
            self.chat_sessions[session_key] = chat_session
        
        return self.chat_sessions[session_key]
    
    async def _get_session_id_for_checkpointer(
        self,
        user_id: str,
        profile_id: str,
        session_id: Optional[str] = None
    ) -> str:
        """Get the session ID to use for LangGraph checkpointer."""
        chat_session = await self._get_or_create_chat_session(user_id, profile_id, session_id)
        return chat_session.session_id
    
    async def _save_to_chat_history(
        self,
        user_id: str,
        profile_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save message to chat history for long-term memory."""
        try:
            chat_session = await self._get_or_create_chat_session(user_id, profile_id, session_id)
            
            # Generate embedding for semantic search
            embedding = None
            try:
                embeddings = get_embeddings()
                embedding = await embeddings.aembed_query(content)
            except Exception as e:
                logger.warning(f"Could not generate embedding: {e}")
            
            # Add message to chat history
            await chat_session.add_message(
                role=role,
                content=content,
                embedding=embedding,
                metadata=metadata or {}
            )
            
            logger.info(f"Saved {role} message to chat history for session {session_id}")
            
        except Exception as e:
            # Suppress verbose errors for terminal sessions
            if "terminal_user" not in user_id:
                logger.error(f"Error saving to chat history: {e}")
            else:
                logger.debug(f"Chat history unavailable: {e}")
    
    async def _get_memory_context(
        self,
        user_id: str,
        profile_id: str,
        session_id: str,
        query: str
    ) -> Dict[str, Any]:
        """Get relevant memory context from chat history."""
        try:
            chat_session = await self._get_or_create_chat_session(user_id, profile_id, session_id)
            
            # Generate embedding for query
            query_embedding = None
            try:
                embeddings = get_embeddings()
                query_embedding = await embeddings.aembed_query(query)
            except Exception as e:
                logger.warning(f"Could not generate query embedding: {e}")
            
            # Get memory context
            memory_context = await chat_session.get_memory_context(
                query_embedding=query_embedding,
                max_similar=3,
                max_recent=5
            )
            
            return memory_context
            
        except Exception as e:
            logger.error(f"Error getting memory context: {e}")
            return {"similar_messages": [], "recent_context": [], "current_session": [], "current_session_context": []}
    
    async def _generate_jql_from_natural_language(self, query: str) -> str:
        """Generate JQL query from natural language using AI."""
        try:
            system_message = SystemMessage(content="""
            You are a JQL (Jira Query Language) expert. Convert natural language queries into valid JQL.
            
            IMPORTANT: 
            1. Do NOT include LIMIT in the JQL query. LIMIT is handled by the API parameters, not the JQL itself.
            2. Do NOT use ORDER BY with date fields like startDate, endDate, or sprint dates. These are not sortable in most JIRA instances.
            3. Only use ORDER BY with standard fields like priority, created, updated, or summary.
            
            Common JQL patterns:
            - assignee = 'username' (for specific assignee)
            - assignee = currentUser() (for current user)
            - issuetype = Bug (for bugs)
            - issuetype = Task (for tasks)
            - issuetype = Story (for stories)
            - sprint = 'sprint_name' (for specific sprint)
            - sprint in openSprints() (for current sprints)
            - sprint in closedSprints() (for closed sprints)
            - status = 'status_name' (for specific status)
            - priority = 'priority_name' (for specific priority)
            - created >= -30d (for recent issues)
            - project in projectsWhereUserHasPermission() (for accessible projects)
            - status = Backlog (for backlog items)
            - ORDER BY priority DESC, created DESC (for sorting)
            
            Examples:
            - "bugs assigned to Orhan" → "issuetype = Bug AND assignee = 'orhan.ors'"
            - "tasks in alt-J sprint" → "issuetype = Task AND sprint = 'alt-J'"
            - "high priority bugs" → "issuetype = Bug AND priority in (Highest, High)"
            - "my open tasks" → "assignee = currentUser() AND issuetype = Task AND status != Done"
            - "top 5 items in the backlog" → "project in projectsWhereUserHasPermission() AND status = Backlog ORDER BY priority DESC, created DESC"
            - "recent bugs" → "issuetype = Bug ORDER BY created DESC"
            - "sprint issues" → "sprint in openSprints()"
            - "closed sprint issues" → "sprint in closedSprints()"
            
            Return ONLY the JQL query, nothing else. If the query is unclear, default to "created >= -30d".
            """)
            
            human_message = HumanMessage(content=f"Convert this query to JQL: {query}")
            
            response = await self.llm.ainvoke([system_message, human_message])
            self._track_llm_cost(response)
            jql_query = response.content.strip()
            
            # Validate the JQL query
            if not jql_query or jql_query.lower() in ['none', 'n/a', 'unknown']:
                logger.warning(f"AI generated invalid JQL for query: {query}")
                return "created >= -30d"  # Default fallback
            
            logger.info(f"Generated JQL for '{query}': {jql_query}")
            return jql_query
            
        except Exception as e:
            logger.error(f"Error generating JQL: {e}")
            return "created >= -30d"  # Default fallback
    
    async def _initialize_jira_mcp(self):
        """Initialize JIRA MCP client."""
        if not self.jira_mcp_path:
            logger.warning("JIRA MCP path not provided, JIRA tools will be disabled")
            return
            
        try:
            # Check if we have JIRA environment variables (following Atlassian MCP documentation)
            jira_url = os.getenv("JIRA_URL")
            jira_username = os.getenv("JIRA_USERNAME") 
            jira_api_token = os.getenv("JIRA_API_TOKEN") or os.getenv("JIRA_API_KEY")  # Fallback for compatibility
            
            if jira_url and jira_username and jira_api_token:
                logger.info(f"JIRA configuration found for: {jira_url}")
                
                # Initialize MCP client with JIRA server
                # Handle Docker-based MCP servers
                if "docker" in self.jira_mcp_path.lower():
                    # Parse Docker command
                    docker_parts = self.jira_mcp_path.split()
                    server_params = StdioServerParameters(
                        command=docker_parts[0],  # "docker"
                        args=docker_parts[1:],     # ["run", "--rm", "--env-file", ".env", "sooperset/mcp-atlassian"]
                        env={
                            "JIRA_URL": jira_url,
                            "JIRA_USERNAME": jira_username,
                            "JIRA_API_TOKEN": jira_api_token,
                            **dict(os.environ)  # Include existing environment
                        }
                    )
                else:
                    # Handle direct executable MCP servers
                    server_params = StdioServerParameters(
                        command="node",  # or "python" depending on the MCP server
                        args=[self.jira_mcp_path],
                        env={
                            "JIRA_URL": jira_url,
                            "JIRA_USERNAME": jira_username,
                            "JIRA_API_TOKEN": jira_api_token,
                            **dict(os.environ)  # Include existing environment
                        }
                    )
                
                # Create MCP client session with stdio transport
                self.jira_stdio_context = stdio_client(server_params)
                read_stream, write_stream = await self.jira_stdio_context.__aenter__()
                
                # Create the client session
                self.jira_session = ClientSession(read_stream, write_stream)
                await self.jira_session.__aenter__()
                
                # Initialize the MCP session
                result = await self.jira_session.initialize()
                logger.info(f"MCP initialized: {result}")
                
                # Get available tools
                self.jira_tools = await self.jira_session.list_tools()
                logger.info(f"JIRA MCP initialized with {len(self.jira_tools.tools)} tools")
                
                self.jira_client = {
                    "session": self.jira_session,
                    "stdio_context": self.jira_stdio_context,
                    "tools": self.jira_tools.tools,
                    "configured": True
                }
                
            else:
                logger.warning("JIRA credentials not found")
                self.jira_client = None
            
        except Exception as e:
            logger.error(f"Failed to initialize JIRA MCP client: {e}")
            logger.info("Falling back to simulation mode")
            # Fallback to simulation
            self.jira_client = {
                "simulation": True,
                "configured": True
            } if jira_url and jira_username and jira_api_token else None
    
    async def route_query(self, state: C9SAgentState) -> C9SAgentState:
        """Route the query to appropriate tools based on content analysis."""
        state["current_step"] = "routing"
        
        try:
            query = state["query"].lower()
            logger.info(f"🛣️ Routing query (lowercase): {query}")
            
            # Define keyword categories
            crypto_keywords = ["cryptobooks", "crypto books", "cryptocurrency", "bitcoin", "ethereum", "blockchain", "defi", "nft"]
            jira_keywords = ["jira", "ticket", "issue", "bug", "bugs", "task", "project", "sprint", "assignee", "assigned"]
            web_keywords = ["search", "find", "latest", "news", "current"]
            
            # Check for keyword matches
            has_crypto = any(keyword in query for keyword in crypto_keywords)
            has_jira = any(keyword in query for keyword in jira_keywords)
            has_web = any(keyword in query for keyword in web_keywords)
            
            # Smart routing - support multiple tools for comprehensive queries
            tools_to_call = []
            
            if has_jira:
                tools_to_call.append("jira_search")
                logger.info(f"🛣️ JIRA keywords detected - will call jira_search")
            
            if has_crypto:
                tools_to_call.append("document_search")
                logger.info(f"🛣️ Crypto keywords detected - will call document_search")
            
            if has_web and not has_crypto:  # Only add web if not crypto (crypto prioritizes docs)
                tools_to_call.append("web_search")
                logger.info(f"🛣️ Web keywords detected - will call web_search")
            
            # Set the routing action based on detected tools
            if len(tools_to_call) > 1:
                state["next_action"] = "parallel_search"
                state["tools_to_call"] = tools_to_call
                logger.info(f"🛣️ Multiple tools detected, routing to parallel execution: {tools_to_call}")
            elif len(tools_to_call) == 1:
                state["next_action"] = tools_to_call[0]
                logger.info(f"🛣️ Single tool detected, routing to: {tools_to_call[0]}")
            else:
                # Use LLM to make routing decision
                routing_prompt = f"""
                Analyze this query and determine the best tool to use:
                Query: {state['query']}
                
                Available tools:
                - jira_search: For JIRA-related queries (tickets, issues, projects)
                - web_search: For general web search and current information
                - direct_answer: For simple questions that don't require external tools
                
                Respond with just the tool name.
                """
                
                response = await self.llm.ainvoke([HumanMessage(content=routing_prompt)])
                self._track_llm_cost(response)
                action = response.content.strip().lower()
                
                if action in ["jira_search", "web_search", "direct_answer"]:
                    state["next_action"] = action
                else:
                    state["next_action"] = "web_search"  # Default fallback
            
            logger.info(f"Routed query to: {state['next_action']}")
            
        except Exception as e:
            logger.error(f"Error in routing: {e}")
            state["next_action"] = "web_search"  # Fallback
            
        return state
    
    async def web_search(self, state: C9SAgentState) -> C9SAgentState:
        """Perform web search using Tavily."""
        state["current_step"] = "web_search"
        
        try:
            logger.info(f"Performing web search for: {state['query']}")
            
            response = self.tavily_client.search(
                query=state["query"],
                search_depth="advanced",
                max_results=5
            )
            
            state["web_results"] = response.get("results", [])
            logger.info(f"Retrieved {len(state['web_results'])} web results")
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            state["web_results"] = []
            
        return state
    
    async def document_search(self, state: C9SAgentState) -> C9SAgentState:
        """Search relevant documents using vector similarity."""
        state["current_step"] = "document_search"
        
        try:
            logger.info(f"Searching documents for: {state['query']}")
            
            # Generate embedding for the query
            embeddings = get_embeddings()
            query_embedding = await embeddings.aembed_query(state["query"])
            
            # Check if this is a crypto-related query for prioritization
            query_lower = state["query"].lower()
            crypto_keywords = ["cryptobooks", "crypto books", "cryptocurrency", "bitcoin", "ethereum", "blockchain", "defi", "nft"]
            is_crypto_query = any(keyword in query_lower for keyword in crypto_keywords)
            
            if is_crypto_query:
                # For crypto queries, use a lower threshold to get more results and prioritize them
                logger.info("🔍 Crypto query detected - using enhanced document search")
                document_results = await vector_search(
                    embedding=query_embedding,
                    limit=10,  # Get more results for crypto queries
                    threshold=0.7,  # Lower threshold to catch more relevant docs
                    source_filter=DocumentSource.INTERCOM_ARTICLE
                )
            else:
                # Standard document search for non-crypto queries
                document_results = await vector_search(
                    embedding=query_embedding,
                    limit=5,
                    threshold=0.8,
                    source_filter=DocumentSource.INTERCOM_ARTICLE
                )
            
            state["document_results"] = document_results
            logger.info(f"Retrieved {len(document_results)} relevant documents")
            
            # Log document titles for debugging
            if document_results:
                titles = [doc.get("title", "Untitled") for doc in document_results]
                logger.info(f"Document titles: {titles}")
                
                if is_crypto_query:
                    logger.info("📚 Crypto query - documents will be prioritized in synthesis")
            
        except Exception as e:
            # Suppress verbose errors for terminal sessions
            if "terminal_user" not in state.get("user_id", ""):
                logger.error(f"Error in document search: {e}")
            else:
                logger.debug(f"Document search unavailable: {e}")
            state["document_results"] = []
            
        return state
    
    async def parallel_search_coordinator(self, state: C9SAgentState) -> C9SAgentState:
        """Coordinate parallel execution of multiple search tools."""
        state["current_step"] = "parallel_search"
        
        try:
            tools_to_call = state.get("tools_to_call", [])
            logger.info(f"🔀 Starting parallel search with tools: {tools_to_call}")
            
            # Execute tools in parallel using asyncio.gather
            tasks = []
            
            if "jira_search" in tools_to_call:
                tasks.append(self.jira_search(state.copy()))
            
            if "document_search" in tools_to_call:
                tasks.append(self.document_search(state.copy()))
                
            if "web_search" in tools_to_call:
                tasks.append(self.web_search(state.copy()))
            
            # Execute all tasks in parallel
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Merge results from all parallel executions
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Error in parallel search: {result}")
                        continue
                    
                    # Merge the results into the main state
                    if result.get("jira_results"):
                        state["jira_results"] = result["jira_results"]
                    if result.get("document_results"):
                        state["document_results"] = result["document_results"]
                    if result.get("web_results"):
                        state["web_results"] = result["web_results"]
                
                logger.info(f"🔀 Parallel search completed - merged results from {len([r for r in results if not isinstance(r, Exception)])} tools")
            
        except Exception as e:
            logger.error(f"Error in parallel search coordination: {e}")
            # Initialize empty results as fallback
            if "jira_search" in tools_to_call and not state.get("jira_results"):
                state["jira_results"] = []
            if "document_search" in tools_to_call and not state.get("document_results"):
                state["document_results"] = []
            if "web_search" in tools_to_call and not state.get("web_results"):
                state["web_results"] = []
        
        return state
    
    async def jira_search(self, state: C9SAgentState) -> C9SAgentState:
        """Search JIRA using MCP tools with smart tool selection."""
        state["current_step"] = "jira_search"
        
        logger.info(f"🔍 Starting JIRA search for query: {state['query']}")
        
        if not self.jira_client:
            await self._initialize_jira_mcp()
        
        if not self.jira_client:
            logger.warning("JIRA MCP client not available")
            state["jira_results"] = []
            return state
        
        # Check for sprint queries and handle them with proper JIRA MCP tools
        query_lower = state["query"].lower()
        sprint_keywords = ["sprint", "latest sprint", "current sprint", "active sprint", "name of latest sprint", "latest sprint name", "sprint details", "last sprint", "sprint information"]
        has_sprint_keywords = any(keyword in query_lower for keyword in sprint_keywords)
        
        if has_sprint_keywords:
            logger.info("Sprint query detected, using JIRA MCP sprint tools")
            sprint_results = await self._get_real_sprint_data(state["query"])
            if sprint_results:
                state["jira_results"] = sprint_results
                logger.info(f"Retrieved {len(sprint_results)} sprint results from JIRA MCP")
                return state
            else:
                logger.warning("Failed to get sprint data from JIRA MCP, continuing with standard search")
                # Fall through to standard JIRA search instead of simulation
        
        try:
            logger.info(f"Searching JIRA for: {state['query']}")
            
            if self.jira_client and self.jira_client.get("configured"):
                # Check if we're using real MCP or simulation
                if self.jira_client.get("simulation"):
                    logger.info("Using JIRA simulation mode")
                    # Fallback simulation code here
                    state["jira_results"] = self._simulate_jira_results(state["query"])
                else:
                    logger.info("Using real JIRA MCP client")
                    logger.info(f"JIRA client config: {self.jira_client}")
                    # Use real MCP client
                    session = self.jira_client["session"]
                    tools = self.jira_client["tools"]
                    
                    logger.info(f"🔧 Available tools: {[tool.name for tool in tools]}")
                    
                    # Smart tool selection based on query analysis
                    selected_tools = await self._select_appropriate_tools(state["query"], tools)
                    logger.info(f"🎯 Selected tools: {selected_tools}")
                    
                    # Try tools in order of preference
                    for tool_name, strategy in selected_tools:
                        try:
                            logger.info(f"🔧 Trying tool: {tool_name} with strategy: {strategy}")
                            result = await self._execute_tool_with_strategy(session, tool_name, strategy, state["query"])
                            
                            if result and result.get("success", False):
                                state["jira_results"] = result.get("data", [])
                                logger.info(f"✅ Success with tool {tool_name}: {len(state['jira_results'])} results")
                                return state
                            else:
                                logger.warning(f"❌ Tool {tool_name} failed: {result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            logger.error(f"❌ Error with tool {tool_name}: {e}")
                            continue
                    
                    # If all tools failed, try fallback strategies
                    logger.info("🔄 All primary tools failed, trying fallback strategies")
                    fallback_result = await self._try_fallback_strategies(session, tools, state["query"])
                    
                    if fallback_result:
                        state["jira_results"] = fallback_result
                        logger.info(f"✅ Fallback successful: {len(fallback_result)} results")
                    else:
                        # No results found - trigger human-in-the-loop for query refinement
                        logger.info("❌ No results found, triggering query refinement")
                        state["requires_human_input"] = True
                        state["human_feedback"] = f"No JIRA issues found for your query: '{state['query']}'. The search tried multiple strategies but couldn't find any matching issues.\n\nWould you like to:\n1. Refine your search terms\n2. Check available projects first\n3. Try a different approach\n\nPlease provide an updated query or say 'NO' to stop searching."
                        state["query_refinement_mode"] = True
                        state["original_query"] = state["query"]
                        return state
            else:
                logger.warning("JIRA client not properly configured")
                state["jira_results"] = []
                
        except Exception as e:
            logger.error(f"Error in JIRA search: {e}")
            state["jira_results"] = []
            
        return state
    
    async def _select_appropriate_tools(self, query: str, tools: list) -> list:
        """Select appropriate tools based on query analysis using React/Reflexion pattern."""
        query_lower = query.lower()
        
        # Analyze query intent
        intent_analysis = {
            "project_query": any(word in query_lower for word in ["project", "projects", "list projects"]),
            "bug_query": any(word in query_lower for word in ["bug", "bugs", "error", "issue", "problem"]),
            "task_query": any(word in query_lower for word in ["task", "tasks", "latest tasks", "recent tasks", "current tasks", "active tasks"]),
            "specific_issue": bool(re.search(r'[A-Z]+-\d+', query)),
            "sprint_query": any(word in query_lower for word in ["sprint", "sprints"]),
            "board_query": any(word in query_lower for word in ["board", "boards"]),
            "user_query": any(word in query_lower for word in ["user", "assignee", "assigned to"]),
            "search_query": any(word in query_lower for word in ["search", "find", "look for", "show me", "what are", "latest", "recent", "current"])
        }
        
        logger.info(f"🔍 Query intent analysis: {intent_analysis}")
        
        # Tool selection strategies based on intent
        tool_strategies = []
        
        # 1. Task-related queries (highest priority for task queries)
        if intent_analysis["task_query"]:
            tool_strategies.extend([
                ("jira_search", "Search for tasks with JQL"),
                ("jira_get_project_issues", "Get all issues from projects")
            ])
        
        # 2. Bug/Issue queries
        elif intent_analysis["bug_query"]:
            tool_strategies.extend([
                ("jira_search", "Search for bugs with JQL"),
                ("jira_get_project_issues", "Get all issues from projects")
            ])
        
        # 3. Project-related queries
        elif intent_analysis["project_query"]:
            tool_strategies.extend([
                ("jira_get_all_projects", "List all available projects"),
                ("jira_get_project_issues", "Get issues from specific project")
            ])
        
        # Look for specific issue queries
        if intent_analysis["specific_issue"]:
            tool_strategies.extend([
                ("jira_get_issue", "Get specific issue details"),
                ("jira_search", "Search for specific issue with JQL")
            ])
        
        # Check for sprint-specific task queries (e.g., "alt-J sprint tasks")
        if "sprint" in query_lower and any(word in query_lower for word in ["task", "tasks", "issue", "issues", "ticket", "tickets"]):
            tool_strategies.extend([
                ("jira_search", "Search for issues within specific sprint"),
                ("jira_get_agile_boards", "Get boards to find sprint issues")
            ])
        
        # 4. Sprint queries
        elif intent_analysis["sprint_query"]:
            tool_strategies.extend([
                ("jira_get_agile_boards", "Get boards first"),
                ("jira_get_sprints_from_board", "Get sprints from boards")
            ])
        
        # 5. Board queries
        elif intent_analysis["board_query"]:
            tool_strategies.extend([
                ("jira_get_agile_boards", "Get all boards"),
                ("jira_get_board_issues", "Get issues from boards")
            ])
        
        # 6. User queries
        elif intent_analysis["user_query"]:
            tool_strategies.extend([
                ("jira_search", "Search by assignee"),
                ("jira_get_user_profile", "Get user profile")
            ])
        
        # 7. General search queries
        elif intent_analysis["search_query"]:
            tool_strategies.extend([
                ("jira_search", "General JQL search"),
                ("jira_get_all_projects", "List projects as fallback")
            ])
        
        # Default fallback
        if not tool_strategies:
            tool_strategies.extend([
                ("jira_search", "General search for tasks and issues"),
                ("jira_get_all_projects", "List projects as fallback")
            ])
        
        return tool_strategies
    
    async def _execute_tool_with_strategy(self, session, tool_name: str, strategy: str, query: str) -> dict:
        """Execute a tool with appropriate strategy and parameters."""
        try:
            logger.info(f"🔧 Executing {tool_name} with strategy: {strategy}")
            
            if tool_name == "jira_get_all_projects":
                # Get all projects first
                result = await session.call_tool(tool_name, {"include_archived": False})
                return self._parse_tool_result(result, "projects")
            
            elif tool_name == "jira_search":
                # Generate smart JQL based on query
                jql_query = await self._generate_smart_jql(query)
                logger.info(f"🔍 Generated JQL: {jql_query}")
                
                # Check if this is a specific ticket query
                ticket_match = re.search(r'\b([A-Z]+-\d+)\b', query.upper())
                is_specific_ticket = ticket_match is not None
                
                if is_specific_ticket:
                    # Request detailed fields for specific ticket queries
                    search_params = {
                        "jql": jql_query,
                        "fields": "key,summary,status,assignee,issuetype,priority,created,updated,project,description,comment,attachment",
                        "limit": 1
                    }
                    logger.info(f"🔍 Specific ticket query detected, using detailed fields")
                else:
                    # Standard fields for general searches
                    search_params = {
                        "jql": jql_query,
                        "fields": "key,summary,status,assignee,issuetype,priority,created,updated,project",
                        "limit": 10
                    }
                
                result = await session.call_tool(tool_name, search_params)
                return self._parse_tool_result(result, "search")
            
            elif tool_name == "jira_get_issue":
                # Extract issue key from query
                issue_match = re.search(r'([A-Z]+-\d+)', query.upper())
                if issue_match:
                    issue_key = issue_match.group(1)
                    logger.info(f"🔍 Getting detailed information for issue: {issue_key}")
                    result = await session.call_tool(tool_name, {"issue_key": issue_key})
                    return self._parse_tool_result(result, "issue")
                else:
                    return {"success": False, "error": "No issue key found in query"}
            
            elif tool_name == "jira_get_agile_boards":
                # Get boards first
                result = await session.call_tool(tool_name, {})
                return self._parse_tool_result(result, "boards")
            
            elif tool_name == "jira_get_sprints_from_board":
                # Get sprints from first available board
                boards_result = await session.call_tool("jira_get_agile_boards", {})
                boards_data = self._parse_tool_result(boards_result, "boards")
                
                if boards_data.get("success") and boards_data.get("data"):
                    board_id = boards_data["data"][0].get("id")
                    if board_id:
                        result = await session.call_tool(tool_name, {"board_id": str(board_id), "state": "active"})
                        return self._parse_tool_result(result, "sprints")
                
                return {"success": False, "error": "No boards available"}
            
            else:
                # Generic tool execution
                result = await session.call_tool(tool_name, {})
                return self._parse_tool_result(result, "generic")
                
        except Exception as e:
            logger.error(f"❌ Error executing {tool_name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_smart_jql(self, query: str) -> str:
        """Generate smart JQL that avoids common errors."""
        query_lower = query.lower()
        
        # Extract key information from query
        project_key = None
        issue_type = None
        assignee = None
        status = None
        ticket_key = None
        
        # Look for specific ticket ID (e.g., CPT-4670, CPT-4671)
        ticket_match = re.search(r'\b([A-Z]+-\d+)\b', query.upper())
        if ticket_match:
            ticket_key = ticket_match.group(1)
            logger.info(f"🔍 Extracted ticket key: {ticket_key}")
        
        # Look for project mentions - be more careful about extraction
        # Only extract if it looks like a real project key (2-5 uppercase letters)
        # But exclude common words that are often mistaken for project keys
        common_words = ["WHAT", "SHOW", "ALL", "THE", "AND", "FOR", "ABOUT", "WITH", "FROM", "INTO", "UPON", "UNDER", "OVER", "BETWEEN", "AMONG", "DURING", "BEFORE", "AFTER", "SINCE", "UNTIL", "WHILE", "WHERE", "WHEN", "WHY", "HOW", "NEXT", "LAST", "CURRENT", "PREVIOUS", "LATEST", "RECENT", "OLDER", "NEWER", "BEST", "WORST", "HIGH", "LOW", "OPEN", "CLOSED", "ACTIVE", "INACTIVE", "PENDING", "RESOLVED", "ASSIGNED", "UNASSIGNED", "BUGS", "TASKS", "ISSUES", "PROJECTS", "FIND"]
        
        project_match = re.search(r'\b([A-Z]{2,5})\b', query.upper())
        if project_match:
            potential_project = project_match.group(1)
            
            if potential_project not in common_words:
                project_key = potential_project
                logger.info(f"🔍 Extracted project key: {project_key}")
            else:
                logger.info(f"🔍 Ignored common word as project key: {potential_project}")
        
        # Look for issue types
        if "bug" in query_lower:
            issue_type = "Bug"
        elif "task" in query_lower:
            issue_type = "Task"
        elif "story" in query_lower:
            issue_type = "Story"
        elif "epic" in query_lower:
            issue_type = "Epic"
        
        # Look for sprint mentions first
        sprint_name = None
        if "alt-j" in query_lower:
            sprint_name = "alt-J"
        
        # Look for assignee mentions
        if "nextgen" in query_lower:
            # This might be a project name, not an assignee
            pass
        elif "orhan" in query_lower:
            # Use the actual name format from JIRA data
            assignee = "Orhan Örs"
        elif "luca" in query_lower and "cardelli" in query_lower:
            # Handle Luca Cardelli
            assignee = "Luca Cardelli"
        
        # Look for status mentions
        if "not done" in query_lower or "not completed" in query_lower:
            status = "not_done"
        elif "done" in query_lower or "completed" in query_lower:
            status = "done"
        elif "code review" in query_lower:
            status = "code_review"
        elif "in progress" in query_lower:
            status = "in_progress"
        
        # Build JQL step by step
        jql_parts = []
        
        # If we have a specific ticket key, prioritize that
        if ticket_key:
            jql_parts.append(f'key = "{ticket_key}"')
        else:
            # For task queries, prioritize recent and active tasks
            if "task" in query_lower or "latest" in query_lower or "recent" in query_lower:
                # Start with recent tasks - be more flexible
                jql_parts.append("created >= -30d")
                
                # Only add task type if explicitly mentioned and no other types
                if "task" in query_lower and not any(word in query_lower for word in ["bug", "story", "epic"]):
                    # But don't force it - let it find any recent issues
                    pass
                
                # Be more flexible with status - don't exclude too many
                # Only exclude very final statuses
                jql_parts.append("status != Closed")
            else:
                # Start with a safe base query - only if we have a valid project
                if project_key:
                    jql_parts.append(f'project = "{project_key}"')
                else:
                    # If no specific project, use a broader search
                    jql_parts.append("created >= -30d")  # Recent issues
            
            # Add issue type filter only if we have a specific type and it's not a general task query
            if issue_type and not ("task" in query_lower and not any(word in query_lower for word in ["bug", "story", "epic"])):
                jql_parts.append(f'issuetype = {issue_type}')
            
            # Add sprint filter (high priority for sprint-specific queries)
            if sprint_name:
                jql_parts.append(f'sprint = "{sprint_name}"')
            
            # Add assignee filter
            if assignee:
                jql_parts.append(f'assignee = "{assignee}"')
            
            # Add status filter based on query
            if status == "not_done":
                jql_parts.append("status != Done")
                jql_parts.append("status != Closed")
            elif status == "done":
                jql_parts.append("status = Done")
            elif status == "code_review":
                jql_parts.append("status = 'Code Review'")
            elif status == "in_progress":
                jql_parts.append("status = 'In Progress'")
            
            # If we have no specific filters, add a broader search
            if len(jql_parts) == 0:
                jql_parts.append("created >= -30d")
        
        # Combine with AND
        jql = " AND ".join(jql_parts)
        
        # Add ordering - for tasks, order by priority and creation date
        if "task" in query_lower:
            jql += " ORDER BY priority DESC, created DESC"
        else:
            jql += " ORDER BY created DESC"
        
        logger.info(f"🔍 Generated smart JQL: {jql}")
        return jql
    
    def _parse_tool_result(self, result, result_type: str) -> dict:
        """Parse tool result and extract data."""
        try:
            if not result or not result.content:
                return {"success": False, "error": "No content in result"}
            
            # Extract content text
            content_text = None
            if hasattr(result.content, 'text'):
                content_text = result.content.text
            elif isinstance(result.content, str):
                content_text = result.content
            elif isinstance(result.content, list) and len(result.content) > 0:
                item = result.content[0]
                if hasattr(item, 'text'):
                    content_text = item.text
                elif isinstance(item, str):
                    content_text = item
            
            if not content_text:
                return {"success": False, "error": "Could not extract content text"}
            
            # Parse JSON
            try:
                parsed = json.loads(content_text)
            except json.JSONDecodeError:
                return {"success": False, "error": f"Could not parse JSON: {content_text[:200]}..."}
            
            # Extract data based on result type
            if result_type == "search":
                if isinstance(parsed, dict) and "issues" in parsed:
                    issues = parsed.get("issues", [])
                    return {
                        "success": True,
                        "data": [self._normalize_issue(issue) for issue in issues],
                        "total": parsed.get("total", 0)
                    }
                else:
                    return {"success": False, "error": "No issues found in search result"}
            
            elif result_type == "projects":
                if isinstance(parsed, list):
                    return {
                        "success": True,
                        "data": [self._normalize_project(project) for project in parsed]
                    }
                else:
                    return {"success": False, "error": "No projects found"}
            
            elif result_type == "issue":
                if isinstance(parsed, dict):
                    return {
                        "success": True,
                        "data": [self._normalize_issue(parsed)]
                    }
                else:
                    return {"success": False, "error": "Invalid issue data"}
            
            elif result_type == "boards":
                if isinstance(parsed, list):
                    return {
                        "success": True,
                        "data": [self._normalize_board(board) for board in parsed]
                    }
                else:
                    return {"success": False, "error": "No boards found"}
            
            elif result_type == "sprints":
                if isinstance(parsed, list):
                    return {
                        "success": True,
                        "data": [self._normalize_sprint(sprint) for sprint in parsed]
                    }
                else:
                    return {"success": False, "error": "No sprints found"}
            
            else:
                # Generic parsing
                return {
                    "success": True,
                    "data": parsed if isinstance(parsed, list) else [parsed]
                }
                
        except Exception as e:
            logger.error(f"Error parsing {result_type} result: {e}")
            return {"success": False, "error": str(e)}
    
    def _normalize_issue(self, issue: dict) -> dict:
        """Normalize issue data structure."""
        normalized = {
            "key": issue.get("key", "N/A"),
            "summary": issue.get("summary", "No summary"),
            "status": issue.get("status", {}).get("name", "Unknown") if isinstance(issue.get("status"), dict) else issue.get("status", "Unknown"),
            "assignee": issue.get("assignee", {}).get("display_name", "Unassigned") if issue.get("assignee") else "Unassigned",
            "type": issue.get("issue_type", {}).get("name", "Unknown") if isinstance(issue.get("issue_type"), dict) else issue.get("type", "Unknown"),
            "priority": issue.get("priority", {}).get("name", "Unknown") if isinstance(issue.get("priority"), dict) else issue.get("priority", "Unknown"),
            "created": issue.get("created", ""),
            "updated": issue.get("updated", ""),
            "project": issue.get("project", {}).get("key", "Unknown") if isinstance(issue.get("project"), dict) else issue.get("project", "Unknown")
        }
        
        # Add detailed information if available
        if issue.get("description"):
            normalized["description"] = issue["description"]
        
        if issue.get("comments"):
            normalized["comments"] = issue["comments"]
        
        if issue.get("attachments"):
            normalized["attachments"] = issue["attachments"]
        
        return normalized
    
    def _normalize_project(self, project: dict) -> dict:
        """Normalize project data structure."""
        return {
            "key": project.get("key", "N/A"),
            "name": project.get("name", "Unknown"),
            "id": project.get("id", "Unknown"),
            "projectTypeKey": project.get("projectTypeKey", "Unknown")
        }
    
    def _normalize_board(self, board: dict) -> dict:
        """Normalize board data structure."""
        return {
            "id": board.get("id", "N/A"),
            "name": board.get("name", "Unknown"),
            "type": board.get("type", "Unknown"),
            "projectKey": board.get("projectKey", "Unknown")
        }
    
    def _normalize_sprint(self, sprint: dict) -> dict:
        """Normalize sprint data structure."""
        return {
            "id": sprint.get("id", "N/A"),
            "name": sprint.get("name", "Unknown"),
            "state": sprint.get("state", "Unknown"),
            "startDate": sprint.get("startDate"),
            "endDate": sprint.get("endDate"),
            "goal": sprint.get("goal", ""),
            "boardId": sprint.get("boardId", "Unknown")
        }
    
    async def _try_fallback_strategies(self, session, tools: list, query: str) -> list:
        """Try fallback strategies when primary tools fail."""
        logger.info("🔄 Trying fallback strategies")
        
        fallback_strategies = [
            ("jira_get_all_projects", "List all projects to understand available data"),
            ("jira_search", "Try with simplified JQL: created >= -30d"),
            ("jira_get_agile_boards", "Get boards to understand structure")
        ]
        
        for tool_name, strategy in fallback_strategies:
            try:
                logger.info(f"🔄 Fallback: {tool_name} - {strategy}")
                result = await self._execute_tool_with_strategy(session, tool_name, strategy, query)
                
                if result.get("success") and result.get("data"):
                    logger.info(f"✅ Fallback {tool_name} succeeded")
                    return result["data"]
                    
            except Exception as e:
                logger.error(f"❌ Fallback {tool_name} failed: {e}")
                continue
        
        return []
    
    def _simulate_jira_results(self, query: str) -> list:
        """Simulate JIRA results for fallback when MCP is not available."""
        query_lower = query.lower()
        logger.info(f"🔧 Simulating JIRA results for query: '{query}'")
        mock_results = []
        
        if any(keyword in query_lower for keyword in ["bug", "error", "issue", "problem"]):
            mock_results.extend([
                {"key": "BUG-123", "summary": "Authentication error in login flow", "status": "Open", "type": "Bug", "assignee": "John Doe"},
                {"key": "BUG-124", "summary": "Database connection timeout", "status": "In Progress", "type": "Bug", "assignee": "Jane Smith"}
            ])
        
        if any(keyword in query_lower for keyword in ["task", "feature", "story"]):
            mock_results.extend([
                {"key": "TASK-456", "summary": "Implement new dashboard features", "status": "To Do", "type": "Task", "assignee": "Orhan Örs"},
                {"key": "STORY-789", "summary": "User profile management", "status": "In Review", "type": "Story", "assignee": "Alice Johnson"}
            ])
        
        if "assigned to" in query_lower and "orhan" in query_lower:
            mock_results = [
                {"key": "TASK-100", "summary": "Setup MCP server integration", "status": "In Progress", "type": "Task", "assignee": "Orhan Örs"},
                {"key": "STORY-200", "summary": "Implement Claude integration", "status": "To Do", "type": "Story", "assignee": "Orhan Örs"},
                {"key": "BUG-300", "summary": "Fix terminal encoding issues", "status": "Done", "type": "Bug", "assignee": "Orhan Örs"}
            ]
        
        if any(keyword in query_lower for keyword in ["high", "priority", "urgent"]):
            mock_results.extend([
                {"key": "URGENT-001", "summary": "Critical security vulnerability", "status": "In Progress", "type": "Bug", "assignee": "Security Team"},
                {"key": "HIGH-002", "summary": "Performance degradation", "status": "Open", "type": "Bug", "assignee": "DevOps Team"}
            ])
        
        # Handle sprint queries
        if any(keyword in query_lower for keyword in ["sprint", "latest sprint", "current sprint", "active sprint", "name of latest sprint", "sprint details", "last sprint", "sprint information"]):
            # Check if user wants multiple sprints
            if any(word in query_lower for word in ["last 2", "last 3", "last 4", "last 5", "multiple", "several"]):
                mock_results = [
                    {
                        "type": "sprint",
                        "name": "Sprint 15 - Q4 2024",
                        "id": "10015",
                        "state": "active",
                        "startDate": "2024-10-15T00:00:00.000+0000",
                        "endDate": "2024-10-29T23:59:59.000+0000",
                        "goal": "Complete CRM integration and fix critical bugs",
                        "board_id": "1001"
                    },
                    {
                        "type": "sprint",
                        "name": "Sprint 14 - Q4 2024",
                        "id": "10014",
                        "state": "closed",
                        "startDate": "2024-10-01T00:00:00.000+0000",
                        "endDate": "2024-10-14T23:59:59.000+0000",
                        "goal": "Implement new dashboard features and bug fixes",
                        "board_id": "1001"
                    }
                ]
            else:
                mock_results = [
                    {
                        "type": "sprint",
                        "name": "Sprint 15 - Q4 2024",
                        "id": "10015",
                        "state": "active",
                        "startDate": "2024-10-15T00:00:00.000+0000",
                        "endDate": "2024-10-29T23:59:59.000+0000",
                        "goal": "Complete CRM integration and fix critical bugs",
                        "board_id": "1001"
                    }
                ]
        
        # If no specific keywords, return general results
        if not mock_results:
            mock_results = [
                {"key": "PROJ-100", "summary": "General project task", "status": "Open", "type": "Task", "assignee": "Team Lead"},
                {"key": "PROJ-101", "summary": "Code review requested", "status": "In Review", "type": "Task", "assignee": "Developer"}
            ]
        
        return mock_results[:5]  # Limit to 5 results
    
    async def _get_real_sprint_data(self, query: str) -> List[Dict[str, Any]]:
        """Get real sprint data using JIRA MCP tools."""
        try:
            if not self.jira_client or not self.jira_client.get("configured"):
                logger.warning("JIRA MCP client not configured for sprint data")
                return []
            
            if self.jira_client.get("simulation"):
                logger.warning("JIRA client in simulation mode, cannot get real sprint data")
                return []
            
            session = self.jira_client["session"]
            tools = self.jira_client["tools"]
            
            logger.info("🔍 Getting real sprint data from JIRA MCP")
            
            # First, get agile boards
            boards_result = await session.call_tool("jira_get_agile_boards", {})
            boards_data = self._parse_tool_result(boards_result, "boards")
            
            if not boards_data.get("success") or not boards_data.get("data"):
                logger.warning("No agile boards found")
                return []
            
            # Get sprints from all available boards and their tasks
            all_results = []
            query_lower = query.lower()
            
            for board in boards_data["data"]:
                board_id = board.get("id")
                if board_id:
                    try:
                        # Get active sprints
                        active_sprints_result = await session.call_tool(
                            "jira_get_sprints_from_board", 
                            {"board_id": str(board_id), "state": "active"}
                        )
                        active_sprints_data = self._parse_tool_result(active_sprints_result, "sprints")
                        
                        if active_sprints_data.get("success") and active_sprints_data.get("data"):
                            for sprint in active_sprints_data["data"]:
                                sprint_name = sprint.get("name", "").lower()
                                
                                # Check if this is the sprint the user is asking about
                                if "alt-j" in query_lower and "alt-j" in sprint_name:
                                    logger.info(f"Found matching alt-J sprint: {sprint.get('name')}")
                                    
                                    # Get issues in this specific sprint using JQL
                                    sprint_jql = f'sprint = "{sprint.get("name")}"'
                                    
                                    try:
                                        sprint_issues_result = await session.call_tool(
                                            "jira_search",
                                            {
                                                "jql": sprint_jql,
                                                "fields": "key,summary,status,assignee,issuetype,priority,created,updated,project",
                                                "limit": 20
                                            }
                                        )
                                        
                                        sprint_issues_data = self._parse_tool_result(sprint_issues_result, "search")
                                        
                                        if sprint_issues_data.get("success") and sprint_issues_data.get("data"):
                                            logger.info(f"Found {len(sprint_issues_data['data'])} issues in {sprint.get('name')}")
                                            # Add the actual issues instead of sprint metadata
                                            for issue in sprint_issues_data["data"]:
                                                issue["sprint_name"] = sprint.get("name")
                                                issue["sprint_state"] = sprint.get("state") 
                                                issue["board_name"] = board.get("name", "Unknown Board")
                                                all_results.append(issue)
                                        else:
                                            logger.warning(f"No issues found in sprint {sprint.get('name')}")
                                    
                                    except Exception as e:
                                        logger.error(f"Error getting issues for sprint {sprint.get('name')}: {e}")
                                
                                # Also add sprint info if it matches general sprint queries
                                elif not ("alt-j" in query_lower) or not ("task" in query_lower):
                                    sprint["type"] = "sprint"
                                    sprint["board_name"] = board.get("name", "Unknown Board")
                                    all_results.append(sprint)
                        
                        # For non-specific queries, also get closed sprints
                        if not ("alt-j" in query_lower and "task" in query_lower):
                            closed_sprints_result = await session.call_tool(
                                "jira_get_sprints_from_board",
                                {"board_id": str(board_id), "state": "closed"}
                            )
                            closed_sprints_data = self._parse_tool_result(closed_sprints_result, "sprints")
                            
                            if closed_sprints_data.get("success") and closed_sprints_data.get("data"):
                                # Only add the most recent closed sprints
                                for sprint in closed_sprints_data["data"][:2]:  # Limit to 2 most recent
                                    sprint["type"] = "sprint"
                                    sprint["board_name"] = board.get("name", "Unknown Board")
                                    all_results.append(sprint)
                                
                    except Exception as e:
                        logger.error(f"Error getting sprints for board {board_id}: {e}")
                        continue
            
            if all_results:
                logger.info(f"Successfully retrieved {len(all_results)} sprint-related results from JIRA MCP")
                return all_results
            else:
                logger.warning("No sprint data found in any boards")
                return []
                
        except Exception as e:
            logger.error(f"Error getting real sprint data: {e}")
            return []
    
    async def check_human_input_needed(self, state: C9SAgentState) -> C9SAgentState:
        """Check if human input is needed before proceeding."""
        state["current_step"] = "human_check"
        
        if not self.enable_human_loop:
            state["requires_human_input"] = False
            return state
        
        try:
            # Simple heuristics for when to ask for human input
            query = state["query"].lower()
            next_action = state.get("next_action")
            
            # Ask for human input if:
            # 1. Query contains sensitive operations
            # 2. Query asks for specific JIRA ticket details
            # 3. Multiple conflicting results
            # 4. Unclear intent
            # 5. NO RESULTS FOUND FROM TOOL CALLS (this is the main case)
            
            sensitive_operations = ["delete", "remove", "close", "assign", "update", "modify"]
            needs_human_input = any(op in query for op in sensitive_operations)
            
            # Check for specific JIRA ticket queries that need human input
            if any(word in query for word in ["jira tool", "tool", "specific", "detail", "hakkında", "about"]):
                needs_human_input = True
            
            # Check if we have NO results from tool calls - but only if tools were actually used
            web_count = len(state.get("web_results", []))
            jira_count = len(state.get("jira_results", []))
            document_count = len(state.get("document_results", []))
            
            # If the router decided on "direct_answer", don't trigger human input for empty tool results
            if next_action == "direct_answer":
                # For direct answers, only trigger human input for sensitive operations or specific requests
                pass  # needs_human_input already set above
            elif web_count == 0 and jira_count == 0 and document_count == 0:
                # Only trigger human input if tools were supposed to be used but returned no results
                if next_action in ["web_search", "jira_search"]:
                    needs_human_input = True  # Tools were used but found no results
            elif jira_count > 0 or document_count > 0:
                # We have JIRA or document results - proceed normally without human input
                needs_human_input = False
            elif web_count > 0:
                # We have web results - proceed normally without human input
                needs_human_input = False
            
            state["requires_human_input"] = needs_human_input
            
            if needs_human_input:
                logger.info(f"Human input required for this query (action: {next_action})")
            else:
                logger.info(f"No human input needed - proceeding with available results (action: {next_action})")
            
        except Exception as e:
            logger.error(f"Error checking human input need: {e}")
            state["requires_human_input"] = False
            
        return state
    
    async def synthesize_response(self, state: C9SAgentState) -> C9SAgentState:
        """Synthesize final response from all available information."""
        state["current_step"] = "synthesis"
        
        try:
            logger.info("Synthesizing final response")
            
            # Prepare context
            context_parts = []
            
            # Track data sources and their reliability
            data_sources = []
            confidence_levels = []
            
            # Check if this is a crypto query to prioritize document results
            query_lower = state["query"].lower()
            crypto_keywords = ["cryptobooks", "crypto books", "cryptocurrency", "bitcoin", "ethereum", "blockchain", "defi", "nft"]
            is_crypto_query = any(keyword in query_lower for keyword in crypto_keywords)
            
            # For crypto queries, prioritize documents first
            if is_crypto_query and state.get("document_results"):
                # Prioritize documents for crypto queries - add them first and with higher weight
                document_context = "\n".join([
                    f"Document: {result.get('title', 'Untitled')} | Source: {result.get('source', 'Unknown')} | Similarity: {result.get('similarity', 0):.3f}\nContent: {result.get('content', '')}"
                    for result in state["document_results"][:3]  # Use full content for crypto queries
                ])
                context_parts.append(f"📚 PRIMARY SOURCE - Knowledge Base Documents (Crypto/Finance):\n{document_context}")
                data_sources.append("knowledge_base_priority")
                confidence_levels.append(0.98)  # Very high confidence for crypto knowledge base
                
                # Add web results as secondary for crypto queries
                if state.get("web_results"):
                    web_context = "\n".join([
                        f"Web: {result.get('title', 'Unknown')} - {result.get('content', '')[:200]}..."
                        for result in state["web_results"][:2]  # Fewer web results for crypto queries
                    ])
                    context_parts.append(f"🌐 SECONDARY SOURCE - Web Search Results:\n{web_context}")
                    data_sources.append("web_search_secondary")
                    confidence_levels.append(0.7)  # Lower confidence for web when we have docs
            else:
                # Standard prioritization for non-crypto queries
                if state.get("web_results"):
                    web_context = "\n".join([
                        f"Web: {result.get('title', 'Unknown')} - {result.get('content', '')[:300]}..."
                        for result in state["web_results"][:3]
                    ])
                    context_parts.append(f"Web Search Results:\n{web_context}")
                    data_sources.append("web_search")
                    confidence_levels.append(0.8)  # Web search is generally reliable
                
                if state.get("document_results"):
                    document_context = "\n".join([
                        f"Document: {result.get('title', 'Untitled')} | Source: {result.get('source', 'Unknown')} | Similarity: {result.get('similarity', 0):.3f}\nContent: {result.get('content', '')[:1000]}..."
                        for result in state["document_results"][:3]
                    ])
                    context_parts.append(f"Relevant Documents:\n{document_context}")
                    data_sources.append("knowledge_base")
                    confidence_levels.append(0.95)  # High confidence for knowledge base documents
            
            if state.get("jira_results"):
                # Check if we have sprint results
                sprint_results = [r for r in state["jira_results"] if r.get("type") == "sprint"]
                ticket_results = [r for r in state["jira_results"] if r.get("type") != "sprint"]
                
                if sprint_results:
                    sprint_context = "\n".join([
                        f"Sprint: {result.get('name', 'Unknown')} | State: {result.get('state', 'Unknown')} | Start: {result.get('startDate', 'Unknown')} | End: {result.get('endDate', 'Unknown')} | Goal: {result.get('goal', 'No goal set')}"
                        for result in sprint_results
                    ])
                    context_parts.append(f"Sprint Information:\n{sprint_context}")
                    data_sources.append("jira_sprint")
                    # Check if this is simulation data
                    if any("2024" in str(result.get('startDate', '')) for result in sprint_results):
                        confidence_levels.append(0.3)  # Low confidence for simulation data
                    else:
                        confidence_levels.append(0.9)  # High confidence for real data
                
                if ticket_results:
                    jira_context = "\n".join([
                        f"JIRA Ticket {result.get('key', 'N/A')}: {result.get('summary', 'No summary')} | Status: {result.get('status', 'Unknown')} | Assignee: {result.get('assignee', 'Unassigned')} | Type: {result.get('type', 'Unknown')} | Priority: {result.get('priority', 'Unknown')}"
                        for result in ticket_results[:5]
                    ])
                    context_parts.append(f"JIRA Results:\n{jira_context}")
                    data_sources.append("jira_tickets")
                    confidence_levels.append(0.9)  # JIRA tickets are generally reliable
                    
                    # Add detailed information for specific ticket queries
                    ticket_match = re.search(r'\b([A-Z]+-\d+)\b', state["query"].upper())
                    if ticket_match and ticket_results:
                        detailed_ticket = ticket_results[0]  # Get the first (and likely only) result
                        
                        # Add ticket details to context
                        ticket_details = f"""
Ticket Details for {detailed_ticket.get('key', 'N/A')}:
- Summary: {detailed_ticket.get('summary', 'No summary')}
- Status: {detailed_ticket.get('status', 'Unknown')}
- Assignee: {detailed_ticket.get('assignee', 'Unassigned')}
- Type: {detailed_ticket.get('type', 'Unknown')}
- Priority: {detailed_ticket.get('priority', 'Unknown')}
- Created: {detailed_ticket.get('created', 'Unknown')}
- Updated: {detailed_ticket.get('updated', 'Unknown')}
- Project: {detailed_ticket.get('project', 'Unknown')}
"""
                        context_parts.append(f"Detailed Ticket Information:\n{ticket_details}")
                        
                        # Note about additional information
                        context_parts.append("Note: For detailed descriptions and comments, you may need to access JIRA directly or use specific JIRA tools.")
            
            if state.get("human_feedback"):
                context_parts.append(f"Human Feedback: {state['human_feedback']}")
                data_sources.append("human_input")
                confidence_levels.append(1.0)  # Human input is most reliable
                
                # If human feedback asks for specific ticket details, try to get them
                ticket_match = re.search(r'([A-Z]+-\d+)', state.get("human_feedback", ""))
                if ticket_match:
                    ticket_key = ticket_match.group(1)
                    logger.info(f"Human feedback requests details for ticket: {ticket_key}")
                    context_parts.append(f"Human requested specific details for ticket: {ticket_key}")
            
            # Add memory context if available
            if state.get("memory_context"):
                memory_context = state["memory_context"]
                
                # Add current session context first (most important for continuity)
                if memory_context.get("current_session_context"):
                    current_context_messages = memory_context["current_session_context"]
                    if current_context_messages:
                        conversation_context = "\n".join([
                            f"{msg.get('role', 'User').upper()}: {msg.get('content', '')}"
                            for msg in current_context_messages
                        ])
                        context_parts.insert(0, f"IMMEDIATE CONVERSATION CONTEXT (Continue from this):\n{conversation_context}")
                        data_sources.append("current_session")
                        confidence_levels.append(1.0)  # Maximum confidence for immediate conversation context
                        logger.info(f"Including immediate conversation context with {len(current_context_messages)} messages")
                elif memory_context.get("current_session"):
                    # Fallback to old method if new field not available
                    current_messages = memory_context["current_session"]
                    if len(current_messages) > 1:
                        conversation_context = "\n".join([
                            f"{msg.get('role', 'User').upper()}: {msg.get('content', '')}"
                            for msg in current_messages[:-1]  # Exclude current query
                        ])
                        context_parts.insert(0, f"Current Conversation:\n{conversation_context}")
                        data_sources.append("current_session")
                        confidence_levels.append(0.9)  # High confidence for current session
                
                # Add similar past messages from other sessions
                if memory_context.get("similar_messages"):
                    similar_context = "\n".join([
                        f"Similar past conversation: {msg.get('content', '')[:200]}..."
                        for msg in memory_context["similar_messages"][:2]
                    ])
                    context_parts.append(f"Relevant Past Conversations:\n{similar_context}")
                    data_sources.append("chat_history")
                    confidence_levels.append(0.7)  # Chat history is moderately reliable
                
                # Add recent context from other sessions (fixed field name)
                if memory_context.get("recent_context"):
                    recent_context = "\n".join([
                        f"Recent conversation: {msg.get('content', '')[:200]}..."
                        for msg in memory_context["recent_context"][:3]
                    ])
                    context_parts.append(f"Recent Conversation History:\n{recent_context}")
            
            # Check for context-dependent short responses (numbers, yes/no, etc.)
            query_lower = state["query"].lower().strip()
            is_short_response = len(query_lower) <= 5 and (
                query_lower.isdigit() or  # Numbers like "1", "2", "3"
                query_lower in ["yes", "no", "y", "n", "ok", "sure", "good", "bad"] or
                any(char.isdigit() for char in query_lower)  # Mixed like "2a", "1b"
            )
            
            if is_short_response and state.get("memory_context", {}).get("current_session_context"):
                # For short responses, emphasize the immediate context even more
                context_parts.insert(0, f"⚠️ CRITICAL: User provided short response '{state['query']}' - This likely refers to options or questions from the IMMEDIATE conversation above. Respond accordingly to what the user is selecting/answering.")
                logger.info(f"Detected short response '{state['query']}' - emphasizing conversation context")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Determine overall confidence and data source awareness
            overall_confidence = min(confidence_levels) if confidence_levels else 0.5
            is_simulation_data = any("2024" in str(result.get('startDate', '')) for result in state.get("jira_results", []) if result.get("type") == "sprint")
            
            # Create synthesis prompt with improved reasoning instructions
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            system_message = SystemMessage(content=f"""
            You are a helpful assistant that provides direct, user-focused answers.
            Current date: {current_date}
            
            CRITICAL INSTRUCTIONS:
            1. PROVIDE DIRECT ANSWERS - Don't overcomplicate or overanalyze
            2. If you have data, SHOW IT CLEARLY to the user
            3. Focus on what the user actually asked for
            4. Be concise but informative
            5. If data is limited, say so briefly and show what you have
            6. Don't make excuses or over-explain limitations
            
            CONVERSATION CONTEXT HANDLING:
            - ALWAYS check the "Current Conversation" context first
            - If the user's query is short (like "yes", "no", "ok", etc.), use the conversation context to understand what they're responding to
            - Maintain conversation continuity by referencing previous exchanges
            - If the user is agreeing to something you previously asked, provide the requested information or action
            - If the user is asking for clarification, provide it based on the conversation history
            - NEVER ignore the conversation context - it's the most important source of information
            - If the user says "yes" to a previous question, answer that question
            - If the user asks for more details about something mentioned earlier, provide those details
            
            SPECIAL HANDLING FOR CRYPTO/CRYPTOBOOKS QUERIES:
            - For queries about cryptocurrency, blockchain, cryptobooks, or related topics
            - PRIORITIZE knowledge base documents as PRIMARY SOURCE
            - Use web results only as supplementary information
            - Trust internal documentation over external web sources
            - Provide comprehensive answers from knowledge base first
            
            Response Guidelines:
            1. Start with a direct answer to the user's question
            2. For crypto queries: Lead with knowledge base information, supplement with web if needed
            3. For other queries: Balance all available sources appropriately
            4. If you have JIRA data, present it clearly in a table or list
            5. Only mention data source issues if they significantly impact the answer
            6. Keep the response focused and actionable
            7. For short responses like "yes", provide the information or action that was previously requested
            
            Data Source Analysis:
            - Current Session: Highest reliability for conversation continuity
            - Knowledge Base (PRIMARY for crypto): Highest reliability, especially for crypto/finance topics
            - Web search results: Generally reliable but may be outdated, secondary for crypto queries
            - JIRA tickets: High reliability for current data
            - Sprint data: Check if dates are current (should be 2025, not 2024)
            - Human feedback: Most reliable source
            - Chat history: Moderately reliable for context
            
            If you detect simulation data (e.g., 2024 dates in 2025), briefly mention this but still provide the available information.
            """)
            
            human_message = HumanMessage(content=f"""
            Query: {state['query']}
            
            Available Information:
            {context if context else "No additional information available."}
            
            Data Sources: {', '.join(data_sources)}
            Overall Confidence: {overall_confidence:.2f}
            Is Simulation Data: {is_simulation_data}
            
            IMPORTANT: 
            - If the query is short (like "yes", "no", "ok"), check the "Current Conversation" context to understand what the user is responding to or agreeing with
            - The "Current Conversation" context is the most important information - use it to maintain conversation continuity
            - If the user says "yes", look at what question or request they're agreeing to in the conversation history
            - If the user asks for more details about something mentioned earlier, provide those details
            - NEVER ignore the conversation context - it's your primary source of information
            
            Provide a direct, user-focused answer to the query. If you have data, show it clearly.
            """)
            
            response = await self.llm.ainvoke([system_message, human_message])
            self._track_llm_cost(response)
            state["final_answer"] = response.content
            
            # Save assistant response to chat history
            await self._save_to_chat_history(
                user_id=state["user_id"],
                profile_id=state["profile_id"],
                session_id=state["session_id"],
                role="assistant",
                content=response.content,
                metadata={
                    "source": "c9s_agent",
                    "web_results_count": len(state.get("web_results", [])),
                    "jira_results_count": len(state.get("jira_results", [])),
                    "document_results_count": len(state.get("document_results", [])),
                    "requires_human_input": state.get("requires_human_input", False),
                    "data_sources": data_sources,
                    "confidence_level": overall_confidence,
                    "is_simulation_data": is_simulation_data,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            logger.info("Response synthesized successfully")
            
        except Exception as e:
            logger.error(f"Error synthesizing response: {e}")
            state["final_answer"] = f"I encountered an error while processing your request: {str(e)}"
            
        return state
    
    def _decide_next_step(self, state: C9SAgentState) -> str:
        """Decide the next step based on routing decision."""
        next_action = state.get("next_action")
        
        if next_action == "parallel_search":
            return "parallel_search"
        elif next_action == "jira_search":
            return "jira_search"
        elif next_action == "web_search":
            return "web_search"
        elif next_action == "document_search":
            return "document_search"  # Direct to document search for crypto queries
        elif next_action == "direct_answer":
            return "document_search"  # Even direct answers should check documents
        else:
            return "web_search"  # Default
    
    def _check_human_input_required(self, state: C9SAgentState) -> str:
        """Check if human input is required."""
        if state.get("requires_human_input", False):
            return "human_interrupt"
        else:
            return "synthesize"
    
    def _build_graph(self):
        """Build the LangGraph-compatible workflow."""
        workflow = StateGraph(C9SAgentState)
        
        # Add nodes
        workflow.add_node("route", self.route_query)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("jira_search", self.jira_search)
        workflow.add_node("document_search", self.document_search)
        workflow.add_node("parallel_search", self.parallel_search_coordinator)
        workflow.add_node("human_check", self.check_human_input_needed)
        workflow.add_node("synthesize", self.synthesize_response)
        
        # Set entry point
        workflow.set_entry_point("route")
        
        # Add edges - from route to search nodes based on condition
        workflow.add_conditional_edges(
            "route",
            self._decide_next_step,
            {
                "parallel_search": "parallel_search",  # New parallel search route
                "web_search": "web_search",
                "jira_search": "jira_search", 
                "document_search": "document_search",  # Direct route for crypto queries
                "human_check": "document_search"  # Even direct answers should check documents
            }
        )
        
        # Parallel search goes directly to human check (already has all results)
        workflow.add_edge("parallel_search", "human_check")
        
        # Individual search nodes go to document search first, then to human check
        workflow.add_edge("web_search", "document_search")
        workflow.add_edge("jira_search", "document_search")
        workflow.add_edge("document_search", "human_check")
        
        # Conditional edge for human input
        workflow.add_conditional_edges(
            "human_check",
            self._check_human_input_required,
            {
                "human_interrupt": "synthesize",  # Will interrupt here
                "synthesize": "synthesize"
            }
        )
        
        # Set finish point
        workflow.set_finish_point("synthesize")
        
        # Compile with checkpointer for memory
        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["synthesize"] if self.enable_human_loop else []
        )
    
    async def process_query(
        self,
        query: str,
        user_id: str,
        profile_id: str,
        session_id: Optional[str] = None,
        human_feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a query through the C9S agent.
        
        Args:
            query: The user query
            user_id: User identifier
            profile_id: Profile identifier  
            session_id: Session identifier for memory
            human_feedback: Optional human feedback for human-in-the-loop
            
        Returns:
            Dictionary with the response and metadata
        """
        # Get session ID from chat session (this ensures consistency)
        thread_id = await self._get_session_id_for_checkpointer(user_id, profile_id, session_id)
        config = {
            "configurable": {"thread_id": thread_id},
            "tags": ["c9s-agent", "claude-llm", "jira-mcp"],
            "metadata": {
                "user_id": user_id,
                "profile_id": profile_id,
                "query": query[:100] if len(query) > 100 else query,  # Truncate for tracing
                "has_human_feedback": bool(human_feedback)
            }
        }
        
        # Save user query to chat history FIRST
        await self._save_to_chat_history(
            user_id=user_id,
            profile_id=profile_id,
            session_id=thread_id,
            role="user",
            content=query,
            metadata={
                "source": "c9s_agent",
                "has_human_feedback": bool(human_feedback),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Get memory context from chat history AFTER saving the current query
        memory_context = await self._get_memory_context(user_id, profile_id, thread_id, query)
        
        # Ensure memory context includes current session for immediate context awareness
        if not memory_context.get("current_session_context"):
            try:
                chat_session = await self._get_or_create_chat_session(user_id, profile_id, thread_id)
                current_messages = chat_session.get_current_messages()
                if current_messages:
                    # Include all messages except the current query for better context
                    memory_context["current_session_context"] = current_messages[:-1] if len(current_messages) > 1 else []
                    logger.info(f"Enhanced memory context with {len(memory_context['current_session_context'])} current session messages")
            except Exception as e:
                logger.warning(f"Could not enhance memory context: {e}")
        
        # Initial state with memory context
        initial_state = C9SAgentState(
            query=query,
            user_id=user_id,
            profile_id=profile_id,
            session_id=thread_id,
            messages=[],
            web_results=[],
            jira_results=[],
            document_results=[],
            next_action=None,
            tools_to_call=None,
            requires_human_input=False,
            human_feedback=human_feedback,
            final_answer="",
            current_step="",
            step_results={},
            step_timings={},
            memory_context=memory_context,
            query_refinement_mode=False,
            original_query=None,
            current_jql=None
        )
        
        try:
            # Process through the graph
            logger.info(f"🚀 Executing graph with session ID: {thread_id}")
            logger.info(f"🚀 Initial state keys: {list(initial_state.keys())}")
            final_state = await self.graph.ainvoke(initial_state, config)
            logger.info(f"✅ Graph execution completed for session: {thread_id}")
            logger.info(f"✅ Final state keys: {list(final_state.keys())}")
            
            return {
                "answer": final_state["final_answer"],
                "query": final_state["query"],
                "session_id": thread_id,
                "requires_human_input": final_state.get("requires_human_input", False),
                "metadata": {
                    "web_results_count": len(final_state.get("web_results", [])),
                    "jira_results_count": len(final_state.get("jira_results", [])),
                    "document_results_count": len(final_state.get("document_results", [])),
                    "next_action": final_state.get("next_action"),
                    "step_results": final_state.get("step_results", {}),
                    "step_timings": final_state.get("step_timings", {})
                },
                "sources": {
                    "web_results": final_state.get("web_results", []),
                    "jira_results": final_state.get("jira_results", []),
                    "document_results": final_state.get("document_results", [])
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "query": query,
                "session_id": thread_id,
                "requires_human_input": False,
                "metadata": {"error": str(e)},
                "sources": {"web_results": [], "jira_results": [], "document_results": []}
            }
    
    async def get_human_input_context(self, session_id: str) -> Dict[str, Any]:
        """Get context for human input decision."""
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = await self.graph.aget_state(config)
            
            return {
                "query": state.values.get("query", ""),
                "current_step": state.values.get("current_step", ""),
                "web_results": state.values.get("web_results", []),
                "jira_results": state.values.get("jira_results", []),
                "next_action": state.values.get("next_action", "")
            }
            
        except Exception as e:
            logger.error(f"Error getting human input context: {e}")
            return {}
    
    async def continue_with_human_feedback(
        self,
        session_id: str,
        human_feedback: str,
        user_id: Optional[str] = None,
        profile_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Continue processing with human feedback."""
        try:
            # Get the actual session ID from chat session if user_id and profile_id are provided
            if user_id and profile_id:
                actual_session_id = await self._get_session_id_for_checkpointer(user_id, profile_id, session_id)
            else:
                actual_session_id = session_id
            
            config = {"configurable": {"thread_id": actual_session_id}}
            
            # Get current state
            current_state = await self.graph.aget_state(config)
            logger.info(f"Retrieved state for session {actual_session_id}: {current_state is not None}")
            if not current_state or not current_state.values:
                # If no state found, we can't continue properly
                logger.error(f"No state found for session {actual_session_id}, cannot continue with human feedback")
                return {
                    "answer": "I'm sorry, but I lost the context of our conversation. Could you please repeat your original question?",
                    "session_id": actual_session_id,
                    "metadata": {"error": "No state found for session"}
                }
            
            # Check if human feedback contains a new query (contains JIRA-related keywords)
            feedback_lower = human_feedback.lower()
            jira_keywords = ["jira", "bug", "bugs", "task", "tasks", "issue", "issues", "assigned", "assignee", "orhan", "done", "not done", "status"]
            contains_new_query = any(keyword in feedback_lower for keyword in jira_keywords)
            
            if contains_new_query:
                logger.info(f"Human feedback contains new query: {human_feedback}")
                # Treat human feedback as a new query
                return await self.process_query(
                    query=human_feedback,
                    user_id=user_id or "default_user",
                    profile_id=profile_id or "default_profile",
                    session_id=actual_session_id
                )
            
            # Create updated state with human feedback
            updated_state = current_state.values.copy()
            updated_state["human_feedback"] = human_feedback
            updated_state["requires_human_input"] = False
            
            # Check if we're in query refinement mode
            if updated_state.get("query_refinement_mode", False):
                logger.info("Handling query refinement feedback...")
                # Handle query refinement
                refined_state = await self._handle_query_refinement(updated_state, human_feedback)
                
                # If still requires human input, return the refined state
                if refined_state.get("requires_human_input", False):
                    return {
                        "answer": refined_state["human_feedback"],
                        "session_id": actual_session_id,
                        "requires_human_input": True,
                        "metadata": {"query_refinement": True}
                    }
                
                # Otherwise, continue with the refined results
                final_state = await self.graph.ainvoke(refined_state, config)
            else:
                # Continue execution from current state
                final_state = await self.graph.ainvoke(updated_state, config)
            
            return {
                "answer": final_state["final_answer"],
                "session_id": actual_session_id,
                "metadata": {
                    "human_feedback_provided": True,
                    "step_results": final_state.get("step_results", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error continuing with human feedback: {e}")
            return {
                "answer": f"Error processing human feedback: {str(e)}",
                "session_id": session_id,
                "metadata": {"error": str(e)}
            }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_jira_mcp()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.jira_client:
            try:
                # Close session first
                if self.jira_client.get("session"):
                    await self.jira_client["session"].__aexit__(exc_type, exc_val, exc_tb)
                # Then close stdio context
                if self.jira_client.get("stdio_context"):
                    await self.jira_client["stdio_context"].__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.error(f"Error closing JIRA MCP connection: {e}")
    
    async def _handle_query_refinement(self, state: C9SAgentState, human_feedback: str) -> C9SAgentState:
        """Handle query refinement when JIRA search returns no results."""
        feedback_lower = human_feedback.lower().strip()
        
        # Check if user wants to stop searching
        if feedback_lower in ["no", "n", "stop", "quit", "exit"]:
            state["query_refinement_mode"] = False
            state["requires_human_input"] = False
            state["human_feedback"] = "Search stopped by user request."
            return state
        
        # User provided a new query
        new_query = human_feedback.strip()
        if new_query:
            logger.info(f"User provided refined query: {new_query}")
            
            # Update the query and try again
            state["query"] = new_query
            state["query_refinement_mode"] = False
            state["requires_human_input"] = False
            
            # Generate new JQL for the refined query
            new_jql = await self._generate_jql_from_natural_language(new_query)
            logger.info(f"New JQL generated: {new_jql}")
            
            # Perform the search again with the new query
            return await self._perform_jira_search_with_jql(state, new_jql)
        
        # Invalid feedback
        state["human_feedback"] = "Please provide a refined query or say 'NO' to stop searching."
        state["requires_human_input"] = True
        return state
    
    async def _perform_jira_search_with_jql(self, state: C9SAgentState, jql_query: str) -> C9SAgentState:
        """Perform JIRA search with a specific JQL query."""
        try:
            session = self.jira_client["session"]
            tools = self.jira_client["tools"]
            
            # Find the jira_search tool
            search_tool = None
            for tool in tools:
                if tool.name == "jira_search":
                    search_tool = tool
                    break
            
            if not search_tool:
                logger.warning("No jira_search tool found")
                state["jira_results"] = []
                return state
            
            # Prepare parameters for jira_search tool
            search_params = {
                "jql": jql_query,
                "fields": "key,summary,status,assignee,issuetype,priority,created,updated",
                "limit": 10
            }
            
            logger.info(f"Executing refined JIRA search with JQL: {jql_query}")
            
            # Execute the search
            try:
                result = await session.call_tool(search_tool.name, search_params)
                logger.info("Refined JIRA search completed successfully")
            except Exception as e:
                logger.error(f"Refined JIRA search failed: {e}")
                state["jira_results"] = []
                return state
            
            # Parse the results
            if result.content:
                content_text = None
                if hasattr(result.content, 'text'):
                    content_text = result.content.text
                elif isinstance(result.content, str):
                    content_text = result.content
                elif isinstance(result.content, list) and len(result.content) > 0:
                    item = result.content[0]
                    if hasattr(item, 'text'):
                        content_text = item.text
                    elif isinstance(item, str):
                        content_text = item
                
                if content_text:
                    try:
                        parsed = json.loads(content_text)
                        if isinstance(parsed, dict) and "issues" in parsed:
                            issues = parsed.get("issues", [])
                            total_results = parsed.get("total", 0)
                            
                            if total_results == 0 and len(issues) == 0:
                                # Still no results, ask again
                                state["requires_human_input"] = True
                                state["human_feedback"] = f"Still no JIRA issues found with the refined query. The search used: {jql_query}\n\nWould you like to try another query or say 'NO' to stop searching?"
                                state["query_refinement_mode"] = True
                                state["current_jql"] = jql_query
                                return state
                            
                            # Process the results
                            jira_results = []
                            for issue in issues:
                                if isinstance(issue, dict):
                                    normalized_issue = {
                                        "key": issue.get("key", "N/A"),
                                        "summary": issue.get("summary", "No summary"),
                                        "status": issue.get("status", {}).get("name", "Unknown"),
                                        "assignee": issue.get("assignee", {}).get("display_name", "Unassigned") if issue.get("assignee") else "Unassigned",
                                        "type": issue.get("issue_type", {}).get("name", "Unknown"),
                                        "priority": issue.get("priority", {}).get("name", "Unknown"),
                                        "created": issue.get("created", ""),
                                        "updated": issue.get("updated", "")
                                    }
                                    jira_results.append(normalized_issue)
                            
                            state["jira_results"] = jira_results[:10]
                            logger.info(f"Found {len(jira_results)} results with refined query")
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Could not parse refined search result: {e}")
                        state["jira_results"] = []
                else:
                    state["jira_results"] = []
            else:
                state["jira_results"] = []
                
        except Exception as e:
            logger.error(f"Error in refined JIRA search: {e}")
            state["jira_results"] = []
        
        return state