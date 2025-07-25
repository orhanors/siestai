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
from .langgraph_compat import StateGraphCompat as StateGraph, START, END, MockPostgresCheckpointer as PostgresCheckpointer
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
    
    # Router decision
    next_action: Optional[str]
    
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
        
        # Build the graph
        self.graph = self._build_graph()
    
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
            logger.error(f"Error saving to chat history: {e}")
    
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
            return {"similar_messages": [], "recent_messages": [], "conversation_context": ""}
    
    async def _generate_jql_from_natural_language(self, query: str) -> str:
        """Generate JQL query from natural language using AI."""
        try:
            system_message = SystemMessage(content="""
            You are a JQL (Jira Query Language) expert. Convert natural language queries into valid JQL.
            
            Common JQL patterns:
            - assignee = 'username' (for specific assignee)
            - assignee = currentUser() (for current user)
            - issuetype = Bug (for bugs)
            - issuetype = Task (for tasks)
            - issuetype = Story (for stories)
            - sprint = 'sprint_name' (for specific sprint)
            - sprint in openSprints() (for current sprints)
            - status = 'status_name' (for specific status)
            - priority = 'priority_name' (for specific priority)
            - created >= -30d (for recent issues)
            
            Examples:
            - "bugs assigned to Orhan" → "issuetype = Bug AND assignee = 'orhan.ors'"
            - "tasks in alt-J sprint" → "issuetype = Task AND sprint = 'alt-J'"
            - "high priority bugs" → "issuetype = Bug AND priority in (Highest, High)"
            - "my open tasks" → "assignee = currentUser() AND issuetype = Task AND status != Done"
            
            Return ONLY the JQL query, nothing else. If the query is unclear, default to "created >= -30d".
            """)
            
            human_message = HumanMessage(content=f"Convert this query to JQL: {query}")
            
            response = await self.llm.ainvoke([system_message, human_message])
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
            
            # Simple routing logic based on keywords
            jira_keywords = ["jira", "ticket", "issue", "bug", "bugs", "task", "project", "sprint", "assignee", "assigned"]
            web_keywords = ["search", "find", "latest", "news", "current"]
            
            if any(keyword in query for keyword in jira_keywords):
                state["next_action"] = "jira_search"
                logger.info(f"🛣️ Matched JIRA keywords, routing to jira_search")
            elif any(keyword in query for keyword in web_keywords):
                state["next_action"] = "web_search"
                logger.info(f"🛣️ Matched web search keywords, routing to web_search")
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
    
    async def jira_search(self, state: C9SAgentState) -> C9SAgentState:
        """Search JIRA using MCP tools."""
        state["current_step"] = "jira_search"
        
        logger.info(f"🔍 Starting JIRA search for query: {state['query']}")
        
        if not self.jira_client:
            await self._initialize_jira_mcp()
        
        if not self.jira_client:
            logger.warning("JIRA MCP client not available")
            state["jira_results"] = []
            return state
        
        try:
            logger.info(f"Searching JIRA for: {state['query']}")
            
            if self.jira_client and self.jira_client.get("configured"):
                # Check if we're using real MCP or simulation
                if self.jira_client.get("simulation"):
                    logger.info("Using JIRA simulation mode")
                    # Fallback simulation code here
                    state["jira_results"] = self._simulate_jira_results(state["query"])
                else:
                    # Use real MCP client
                    session = self.jira_client["session"]
                    tools = self.jira_client["tools"]
                    
                    # Find the jira_search tool specifically
                    search_tool = None
                    for tool in tools:
                        if tool.name == "jira_search":
                            search_tool = tool
                            break
                    
                    if search_tool:
                        logger.info(f"Using MCP tool: {search_tool.name}")
                        
                        # Check if this is a specific ticket query
                        query_lower = state["query"].lower()
                        
                        # Look for specific ticket numbers (e.g., CPT-4670)
                        import re
                        ticket_match = re.search(r'([A-Z]+-\d+)', state["query"].upper())
                        
                        if ticket_match:
                            # This is a specific ticket query
                            ticket_key = ticket_match.group(1)
                            logger.info(f"Looking for specific ticket: {ticket_key}")
                            
                            # Use jira_get_issue tool if available
                            get_issue_tool = None
                            for tool in tools:
                                if tool.name == "jira_get_issue":
                                    get_issue_tool = tool
                                    break
                            
                            if get_issue_tool:
                                try:
                                    issue_params = {"issue_key": ticket_key}
                                    issue_result = await session.call_tool(get_issue_tool.name, issue_params)
                                    
                                    if issue_result.content:
                                        # Parse the specific issue result
                                        content_text = None
                                        if hasattr(issue_result.content, 'text'):
                                            content_text = issue_result.content.text
                                        elif isinstance(issue_result.content, str):
                                            content_text = issue_result.content
                                        
                                        if content_text:
                                            try:
                                                parsed_issue = json.loads(content_text)
                                                if isinstance(parsed_issue, dict):
                                                    # Normalize the issue data
                                                    normalized_issue = {
                                                        "key": parsed_issue.get("key", ticket_key),
                                                        "summary": parsed_issue.get("summary", "No summary"),
                                                        "status": parsed_issue.get("status", {}).get("name", "Unknown"),
                                                        "assignee": parsed_issue.get("assignee", {}).get("display_name", "Unassigned") if parsed_issue.get("assignee") else "Unassigned",
                                                        "type": parsed_issue.get("issue_type", {}).get("name", "Unknown"),
                                                        "priority": parsed_issue.get("priority", {}).get("name", "Unknown"),
                                                        "created": parsed_issue.get("created", ""),
                                                        "updated": parsed_issue.get("updated", ""),
                                                        "description": parsed_issue.get("description", ""),
                                                        "comments": parsed_issue.get("comments", []),
                                                        "attachments": parsed_issue.get("attachments", [])
                                                    }
                                                    state["jira_results"] = [normalized_issue]
                                                    logger.info(f"Retrieved specific ticket: {ticket_key}")
                                                    return state
                                            except json.JSONDecodeError as e:
                                                logger.warning(f"Could not parse specific issue result: {e}")
                                
                                except Exception as e:
                                    logger.error(f"Error getting specific issue: {e}")
                        
                        # Generate JQL using AI
                        jql_query = await self._generate_jql_from_natural_language(state["query"])
                        
                        # Prepare parameters for jira_search tool
                        search_params = {
                            "jql": jql_query,
                            "fields": "key,summary,status,assignee,issuetype,priority,created,updated",
                            "limit": 10
                        }
                        
                        logger.info(f"Executing JIRA search with JQL: {jql_query}")
                        
                        # Execute MCP tool
                        try:
                            result = await session.call_tool(search_tool.name, search_params)
                            logger.info(f"JIRA search completed successfully")
                        except Exception as e:
                            logger.error(f"JIRA search failed: {e}")
                            result = None
                        
                        if result.content:
                            # Get the content text regardless of type
                            content_text = None
                            if hasattr(result.content, 'text'):
                                content_text = result.content.text
                            elif hasattr(result.content, 'type') and result.content.type == 'text':
                                content_text = str(result.content)
                            elif isinstance(result.content, str):
                                content_text = result.content
                            elif isinstance(result.content, list):
                                # Handle list of content items
                                for item in result.content:
                                    if hasattr(item, 'text'):
                                        content_text = item.text
                                        break
                                    elif isinstance(item, str):
                                        content_text = item
                                        break
                            
                            if content_text:
                                try:
                                    parsed = json.loads(content_text)
                                    if isinstance(parsed, dict) and "issues" in parsed:
                                        # Extract issues from the response
                                        issues = parsed.get("issues", [])
                                        total_results = parsed.get("total", 0)
                                        
                                        # Check if we got no results and this is a JIRA search
                                        if total_results == 0 and len(issues) == 0:
                                            logger.info(f"No JIRA results found for query: {state['query']}")
                                            logger.info(f"JQL used: {jql_query}")
                                            
                                            # Ask user if they want to refine their query
                                            state["requires_human_input"] = True
                                            state["human_feedback"] = f"No JIRA issues found for your query. The search used: {jql_query}\n\nWould you like to refine your search? Please provide an updated query or say 'NO' to stop searching."
                                            state["query_refinement_mode"] = True
                                            state["original_query"] = state["query"]
                                            state["current_jql"] = jql_query
                                            return state
                                        
                                        for issue in issues:
                                            if isinstance(issue, dict):
                                                # Normalize the issue data
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
                                                state["jira_results"].append(normalized_issue)
                                    elif isinstance(parsed, list):
                                        state["jira_results"].extend(parsed)
                                    else:
                                        state["jira_results"].append(parsed)
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Could not parse MCP result as JSON: {e}")
                                    logger.warning(f"Raw content: {content_text[:200]}...")
                                    logger.warning(f"Content type: {type(result.content)}")
                            else:
                                logger.warning(f"Could not extract text content from MCP result: {type(result.content)}")
                                logger.warning(f"Content: {result.content}")
                            
                            state["jira_results"] = state["jira_results"][:10]  # Limit results
                            logger.info(f"Retrieved {len(state['jira_results'])} JIRA results via MCP")
                        else:
                            logger.warning("No content in MCP result")
                            state["jira_results"] = []
                    else:
                        logger.warning("No suitable JIRA search tool found in MCP")
                        # Fallback to simulation
                        state["jira_results"] = self._simulate_jira_results(state["query"])
            else:
                logger.warning("JIRA client not properly configured")
                state["jira_results"] = []
                
        except Exception as e:
            logger.error(f"Error in JIRA search: {e}")
            state["jira_results"] = []
            
        return state
    
    def _simulate_jira_results(self, query: str) -> list:
        """Simulate JIRA results for fallback when MCP is not available."""
        query_lower = query.lower()
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
        
        # If no specific keywords, return general results
        if not mock_results:
            mock_results = [
                {"key": "PROJ-100", "summary": "General project task", "status": "Open", "type": "Task", "assignee": "Team Lead"},
                {"key": "PROJ-101", "summary": "Code review requested", "status": "In Review", "type": "Task", "assignee": "Developer"}
            ]
        
        return mock_results[:5]  # Limit to 5 results
    
    async def check_human_input_needed(self, state: C9SAgentState) -> C9SAgentState:
        """Check if human input is needed before proceeding."""
        state["current_step"] = "human_check"
        
        if not self.enable_human_loop:
            state["requires_human_input"] = False
            return state
        
        try:
            # Simple heuristics for when to ask for human input
            query = state["query"].lower()
            
            # Ask for human input if:
            # 1. Query contains sensitive operations
            # 2. Query asks for specific JIRA ticket details
            # 3. Multiple conflicting results
            # 4. Unclear intent
            
            sensitive_operations = ["delete", "remove", "close", "assign", "update", "modify"]
            needs_human_input = any(op in query for op in sensitive_operations)
            
            # Check for specific JIRA ticket queries
            if any(word in query for word in ["jira tool", "tool", "specific", "detail", "hakkında", "about"]):
                needs_human_input = True
            
            # Check if we have conflicting or unclear results
            web_count = len(state.get("web_results", []))
            jira_count = len(state.get("jira_results", []))
            
            if web_count == 0 and jira_count == 0:
                needs_human_input = True  # No results found
            
            state["requires_human_input"] = needs_human_input
            
            if needs_human_input:
                logger.info("Human input required for this query")
            
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
            
            if state.get("web_results"):
                web_context = "\n".join([
                    f"Web: {result.get('title', 'Unknown')} - {result.get('content', '')[:300]}..."
                    for result in state["web_results"][:3]
                ])
                context_parts.append(f"Web Search Results:\n{web_context}")
            
            if state.get("jira_results"):
                jira_context = "\n".join([
                    f"JIRA Ticket {result.get('key', 'N/A')}: {result.get('summary', 'No summary')} | Status: {result.get('status', 'Unknown')} | Assignee: {result.get('assignee', 'Unassigned')} | Type: {result.get('type', 'Unknown')} | Priority: {result.get('priority', 'Unknown')}"
                    for result in state["jira_results"][:5]
                ])
                context_parts.append(f"JIRA Results:\n{jira_context}")
            
            if state.get("human_feedback"):
                context_parts.append(f"Human Feedback: {state['human_feedback']}")
                
                # If human feedback asks for specific ticket details, try to get them
                ticket_match = re.search(r'([A-Z]+-\d+)', state.get("human_feedback", ""))
                if ticket_match:
                    ticket_key = ticket_match.group(1)
                    logger.info(f"Human feedback requests details for ticket: {ticket_key}")
                    context_parts.append(f"Human requested specific details for ticket: {ticket_key}")
            
            # Add memory context if available
            if state.get("memory_context"):
                memory_context = state["memory_context"]
                if memory_context.get("similar_messages"):
                    similar_context = "\n".join([
                        f"Similar past conversation: {msg.get('content', '')[:200]}..."
                        for msg in memory_context["similar_messages"][:2]
                    ])
                    context_parts.append(f"Relevant Past Conversations:\n{similar_context}")
                
                if memory_context.get("recent_messages"):
                    recent_context = "\n".join([
                        f"Recent conversation: {msg.get('content', '')[:200]}..."
                        for msg in memory_context["recent_messages"][:3]
                    ])
                    context_parts.append(f"Recent Conversation History:\n{recent_context}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Create synthesis prompt
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            system_message = SystemMessage(content=f"""
            You are a helpful assistant that synthesizes information from multiple sources.
            Current date: {current_date}
            
            Instructions:
            1. Provide a clear, comprehensive answer based on the available information
            2. If using JIRA information, be specific about ticket numbers, statuses, assignees, and priorities
            3. When presenting JIRA results, format them clearly with ticket keys, summaries, and status
            4. If using web search results, cite sources when relevant
            5. If human feedback was provided, incorporate it appropriately
            6. Be concise but thorough
            7. If you don't have enough information, say so clearly
            8. For JIRA queries, always mention the number of tickets found and provide key details
            """)
            
            human_message = HumanMessage(content=f"""
            Query: {state['query']}
            
            Available Information:
            {context if context else "No additional information available."}
            
            Please provide a comprehensive response.
            """)
            
            response = await self.llm.ainvoke([system_message, human_message])
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
                    "requires_human_input": state.get("requires_human_input", False),
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
        
        if next_action == "jira_search":
            return "jira_search"
        elif next_action == "web_search":
            return "web_search"
        elif next_action == "direct_answer":
            return "human_check"
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
        workflow.add_node("human_check", self.check_human_input_needed)
        workflow.add_node("synthesize", self.synthesize_response)
        
        # Set entry point
        workflow.set_entry_point("route")
        
        # Add edges - from route to search nodes based on condition
        workflow.add_conditional_edges(
            "route",
            self._decide_next_step,
            {
                "web_search": "web_search",
                "jira_search": "jira_search",
                "human_check": "human_check"
            }
        )
        
        # Both search nodes go to human check
        workflow.add_edge("web_search", "human_check")
        workflow.add_edge("jira_search", "human_check")
        
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
        
        # Get memory context from chat history using the same session ID
        memory_context = await self._get_memory_context(user_id, profile_id, thread_id, query)
        
        # Save user query to chat history
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
        
        # Initial state with memory context
        initial_state = C9SAgentState(
            query=query,
            user_id=user_id,
            profile_id=profile_id,
            session_id=thread_id,
            messages=[],
            web_results=[],
            jira_results=[],
            next_action=None,
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
                    "next_action": final_state.get("next_action"),
                    "step_results": final_state.get("step_results", {}),
                    "step_timings": final_state.get("step_timings", {})
                },
                "sources": {
                    "web_results": final_state.get("web_results", []),
                    "jira_results": final_state.get("jira_results", [])
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
                "sources": {"web_results": [], "jira_results": []}
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