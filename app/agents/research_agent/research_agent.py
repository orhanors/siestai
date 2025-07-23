"""Research Agent with LangGraph for document retrieval and web search."""

import os
import logging
import time
from datetime import datetime
from typing import TypedDict, List, Optional, Dict, Any, Callable, AsyncGenerator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import SystemMessage, HumanMessage

from app.memory.database.database import vector_search, hybrid_search
from app.memory.knowledge_graph.knowledge_graph import KGClient
from app.memory.history.session_manager import ChatSession, memory_manager
from app.services.embedding_service import get_embeddings

logger = logging.getLogger(__name__)


def track_step(step_name: str):
    """Decorator to track step execution time and results."""
    def decorator(func):
        async def wrapper(self, state: ResearchState) -> ResearchState:
            start_time = time.time()
            state["current_step"] = step_name
            
            # Notify step callback if available
            if hasattr(self, 'step_callback') and self.step_callback:
                await self.step_callback(f"🔄 Starting: {step_name}", state)
            
            try:
                result = await func(self, state)
                
                # Track timing
                duration = time.time() - start_time
                if "step_timings" not in result:
                    result["step_timings"] = {}
                result["step_timings"][step_name] = duration
                
                # Track results
                if "step_results" not in result:
                    result["step_results"] = {}
                
                # Store step-specific results
                if step_name == "retrieve_memory":
                    result["step_results"][step_name] = {
                        "memory_found": bool(result.get("memory_context", {})),
                        "similar_messages": len(result.get("memory_context", {}).get("similar_messages", [])),
                        "recent_messages": len(result.get("memory_context", {}).get("recent_context", []))
                    }
                elif step_name == "retrieve_documents":
                    result["step_results"][step_name] = {
                        "documents_found": len(result.get("documents", [])),
                        "document_titles": [doc.get("title", "Unknown") for doc in result.get("documents", [])]
                    }
                elif step_name == "search_knowledge_graph":
                    result["step_results"][step_name] = {
                        "kg_results_found": len(result.get("kg_results", [])),
                        "kg_entities": [kg.get("name", "Unknown") for kg in result.get("kg_results", [])]
                    }
                elif step_name == "web_search":
                    result["step_results"][step_name] = {
                        "web_results_found": len(result.get("web_results", [])),
                        "web_titles": [web.get("title", "Unknown") for web in result.get("web_results", [])]
                    }
                elif step_name == "synthesize_answer":
                    result["step_results"][step_name] = {
                        "answer_length": len(result.get("final_answer", "")),
                        "sources_used": result.get("metadata", {}).get("sources_used", [])
                    }
                
                # Notify completion
                if hasattr(self, 'step_callback') and self.step_callback:
                    await self.step_callback(f"✅ Completed: {step_name} ({duration:.2f}s)", result)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                if hasattr(self, 'step_callback') and self.step_callback:
                    await self.step_callback(f"❌ Failed: {step_name} - {str(e)}", state)
                raise
                
        return wrapper
    return decorator


class ResearchState(TypedDict):
    """State for the research agent."""
    query: str
    user_id: Optional[str]
    profile_id: Optional[str]
    session_id: Optional[str]
    documents: List[Dict[str, Any]]
    kg_results: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    memory_context: Dict[str, Any]
    final_answer: str
    context: str
    metadata: Dict[str, Any]
    # Step tracking
    current_step: str
    step_results: Dict[str, Any]
    step_timings: Dict[str, float]


class ResearchAgent:
    """LangGraph-based research agent with document retrieval and web search."""
    
    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_documents: int = 5,
        max_web_results: int = 3,
        enable_kg: bool = True,
        enable_web_search: bool = True,
        enable_memory: bool = True,
        step_callback: Optional[Callable[[str, ResearchState], None]] = None
    ):
        """Initialize the research agent.
        
        Args:
            model: OpenAI model to use
            temperature: Model temperature
            max_documents: Maximum documents to retrieve
            max_web_results: Maximum web search results
            enable_kg: Whether to use knowledge graph
            enable_web_search: Whether to use web search
            enable_memory: Whether to use chat history memory
            step_callback: Optional callback function for step progress
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.max_documents = max_documents
        self.max_web_results = max_web_results
        self.enable_kg = enable_kg
        self.enable_web_search = enable_web_search
        self.enable_memory = enable_memory
        self.step_callback = step_callback
        
        # Initialize services
        self.embeddings = get_embeddings()
        
        if self.enable_kg:
            self.kg_client = KGClient()
            
        if self.enable_web_search:
            self.web_search = TavilySearchResults(
                max_results=max_web_results,
                search_depth="advanced"
            )
        
        # Initialize memory manager if enabled
        if self.enable_memory:
            self.memory_manager = memory_manager
        
        # Build the graph
        self.graph = self._build_graph()
    
    @track_step("retrieve_documents")
    async def retrieve_documents(self, state: ResearchState) -> ResearchState:
        """Retrieve relevant documents from the database."""
        try:
            logger.info(f"Retrieving documents for query: {state['query']}")
            
            # Get embedding for the query
            query_embedding = await self.embeddings.aembed_query(state["query"])
            
            # Perform vector search
            documents = await vector_search(
                embedding=query_embedding,
                limit=self.max_documents,
                threshold=0.7
            )
            
            state["documents"] = documents
            logger.info(f"Retrieved {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            state["documents"] = []
            
        return state
    
    @track_step("search_knowledge_graph")
    async def search_knowledge_graph(self, state: ResearchState) -> ResearchState:
        """Search the knowledge graph for relevant information."""
        if not self.enable_kg:
            state["kg_results"] = []
            return state
            
        try:
            logger.info(f"Searching knowledge graph for: {state['query']}")
            
            # Search knowledge graph
            kg_results = await self.kg_client.search(
                query=state["query"],
                limit=5
            )
            
            state["kg_results"] = kg_results
            logger.info(f"Retrieved {len(kg_results)} KG results")
            
        except Exception as e:
            logger.error(f"Error searching knowledge graph: {e}")
            state["kg_results"] = []
            
        return state
    
    @track_step("web_search")
    async def web_search_step(self, state: ResearchState) -> ResearchState:
        """Perform web search using Tavily."""
        if not self.enable_web_search:
            state["web_results"] = []
            return state
            
        try:
            logger.info(f"Performing web search for: {state['query']}")
            
            # Perform web search
            web_results = await self.web_search.ainvoke({"query": state["query"]})
            
            state["web_results"] = web_results
            logger.info(f"Retrieved {len(web_results)} web results")
            
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            state["web_results"] = []
            
        return state
    
    @track_step("retrieve_memory")
    async def retrieve_memory(self, state: ResearchState) -> ResearchState:
        """Retrieve memory context from chat history."""
        if not self.enable_memory or not state.get("user_id") or not state.get("profile_id"):
            state["memory_context"] = {}
            return state
        
        try:
            logger.info(f"Retrieving memory context for query: {state['query']}")
            
            # Get or create chat session
            chat_session = await self.memory_manager.get_or_create_session(
                user_id=state["user_id"],
                profile_id=state["profile_id"],
                session_id=state.get("session_id"),
                similarity_threshold=0.6  # Lower threshold for better recall
            )
            
            # Get query embedding for similarity search
            query_embedding = await self.embeddings.aembed_query(state["query"])
            
            # Get memory context
            memory_context = await chat_session.get_memory_context(
                query_embedding=query_embedding,
                max_similar=3,
                max_recent=5
            )
            
            state["memory_context"] = memory_context
            logger.info(f"Retrieved memory context: {len(memory_context.get('similar_messages', []))} similar, "
                       f"{len(memory_context.get('recent_context', []))} recent")
            
        except Exception as e:
            logger.error(f"Error retrieving memory context: {e}")
            state["memory_context"] = {}
        
        return state
    
    def _should_use_web_search(self, state: ResearchState) -> str:
        """Decide whether to use web search based on document retrieval results."""
        documents = state.get("documents", [])
        kg_results = state.get("kg_results", [])
        
        # Use web search if we have few relevant documents
        if len(documents) < 2 and len(kg_results) < 2:
            return "web_search"
        
        # Or if the query seems to be about recent events
        query_lower = state["query"].lower()
        time_indicators = ["recent", "latest", "current", "today", "2024", "2025", "now"]
        if any(indicator in query_lower for indicator in time_indicators):
            return "web_search"
            
        return "synthesize"
    
    @track_step("synthesize_answer")
    async def synthesize_answer(self, state: ResearchState) -> ResearchState:
        """Synthesize the final answer from all sources."""
        try:
            logger.info("Synthesizing final answer")
            
            # Prepare context from all sources
            context_parts = []
            
            # Add document context
            if state.get("documents"):
                doc_context = "\n\n".join([
                    f"Document: {doc.get('title', 'Unknown')}\n{doc.get('content', '')[:500]}..."
                    for doc in state["documents"][:3]
                ])
                context_parts.append(f"Relevant Documents:\n{doc_context}")
            
            # Add knowledge graph context
            if state.get("kg_results"):
                kg_context = "\n".join([
                    f"KG Entity: {result.get('name', 'Unknown')} - {result.get('description', '')[:200]}..."
                    for result in state["kg_results"][:3]
                ])
                context_parts.append(f"Knowledge Graph:\n{kg_context}")
            
            # Add web search context
            if state.get("web_results"):
                web_context = "\n".join([
                    f"Web Result: {result.get('title', 'Unknown')}\n{result.get('content', '')[:300]}..."
                    for result in state["web_results"][:3]
                ])
                context_parts.append(f"Web Search Results:\n{web_context}")
            
            # Add memory context
            if state.get("memory_context") and self.enable_memory:
                memory_context = state["memory_context"]
                
                # Add similar messages from history
                if memory_context.get("similar_messages"):
                    similar_context = "\n".join([
                        f"Past Message: {msg.get('content', '')[:200]}... (similarity: {msg.get('similarity', 0):.2f})"
                        for msg in memory_context["similar_messages"][:2]
                    ])
                    context_parts.append(f"Similar Past Conversations:\n{similar_context}")
                
                # Add recent activity (prioritize this over similar messages for better continuity)
                if memory_context.get("recent_context"):
                    recent_context = "\n".join([
                        f"{msg.get('role', 'User').upper()}: {msg.get('content', '')}"
                        for msg in memory_context["recent_context"][:5]  # Show more recent context
                    ])
                    context_parts.insert(0, f"Recent Conversation:\n{recent_context}")  # Insert at beginning for priority
            
            context = "\n\n---\n\n".join(context_parts)
            state["context"] = context
            
            # Prepare references for documents with content_url
            document_references = []
            if state.get("documents"):
                for i, doc in enumerate(state["documents"], 1):
                    if doc.get("content_url"):
                        document_references.append({
                            "number": i,
                            "title": doc.get("title", "Unknown Document"),
                            "url": doc.get("content_url"),
                            "source": doc.get("source", "unknown")
                        })
            
            # Get current date for temporal context
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Create the prompt
            memory_instruction = ""
            if self.enable_memory and state.get("memory_context"):
                memory_context = state["memory_context"]
                has_recent = bool(memory_context.get("recent_context"))
                has_similar = bool(memory_context.get("similar_messages"))
                
                if has_recent or has_similar:
                    memory_instruction = f"""
10. IMPORTANT: Use the conversation history provided in the context to maintain continuity
11. When a user asks about information mentioned in recent conversations, refer to that history
12. If the user asks about their name, preferences, or previous topics, check the Recent Conversation section
13. Always prioritize recent conversation context over web search for personal information"""
            
            system_message = SystemMessage(content=f"""You are a research assistant that synthesizes information from multiple sources to provide comprehensive, accurate answers.

Current date: {current_date}

Instructions:
1. Use the provided context from documents, knowledge graph, and web search results
2. Provide a clear, well-structured answer to the user's question
3. Cite your sources when possible
4. If information is conflicting, acknowledge this and explain
5. If you don't have enough information, say so clearly
6. Be concise but thorough
7. When referencing documents from the knowledge base, you can refer to them by their titles
8. Consider the current date when discussing recent events, trends, or time-sensitive information
9. When web search results contain recent information, prioritize it over older documents for current events{memory_instruction}""")
            
            human_message = HumanMessage(content=f"""Question: {state['query']}

Context:
{context}

Please provide a comprehensive answer based on the available information.""")
            
            response = await self.llm.ainvoke([system_message, human_message])
            state["final_answer"] = response.content
            
            # Add metadata
            state["metadata"] = {
                "num_documents": len(state.get("documents", [])),
                "num_kg_results": len(state.get("kg_results", [])),
                "num_web_results": len(state.get("web_results", [])),
                "document_references": document_references,
                "sources_used": [
                    source for source in ["documents", "knowledge_graph", "web_search"]
                    if state.get(f"{source.replace('_', '')}_results" if source != "documents" else source, [])
                ]
            }
            
            logger.info("Answer synthesized successfully")
            
        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            state["final_answer"] = f"I encountered an error while processing your request: {str(e)}"
            state["metadata"] = {"error": str(e)}
            
        return state
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("retrieve_memory", self.retrieve_memory)
        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("search_kg", self.search_knowledge_graph)
        workflow.add_node("web_search", self.web_search_step)
        workflow.add_node("synthesize", self.synthesize_answer)
        
        # Add edges - memory retrieval first
        workflow.add_edge(START, "retrieve_memory")
        workflow.add_edge("retrieve_memory", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "search_kg")
        
        # Conditional edge for web search
        workflow.add_conditional_edges(
            "search_kg",
            self._should_use_web_search,
            {
                "web_search": "web_search",
                "synthesize": "synthesize"
            }
        )
        
        workflow.add_edge("web_search", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    async def research(
        self, 
        query: str, 
        user_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the research agent on a query.
        
        Args:
            query: The research query
            user_id: User identifier for memory context
            profile_id: Profile identifier for memory context
            session_id: Session identifier for memory context
            
        Returns:
            Dictionary with the answer and metadata
        """
        initial_state = ResearchState(
            query=query,
            user_id=user_id,
            profile_id=profile_id,
            session_id=session_id,
            documents=[],
            kg_results=[],
            web_results=[],
            memory_context={},
            final_answer="",
            context="",
            metadata={},
            current_step="",
            step_results={},
            step_timings={}
        )
        
        try:
            # Track last user info for session cleanup
            if user_id and profile_id:
                self._last_user_id = user_id
                self._last_profile_id = profile_id
                self._last_session_id = session_id
            
            final_state = await self.graph.ainvoke(initial_state)
            
            # Store the query and response in chat history if memory is enabled
            if self.enable_memory and user_id and profile_id:
                await self._store_chat_interaction(
                    query=query,
                    response=final_state["final_answer"],
                    user_id=user_id,
                    profile_id=profile_id,
                    session_id=session_id,
                    metadata=final_state["metadata"]
                )
            
            return {
                "answer": final_state["final_answer"],
                "query": final_state["query"],
                "context": final_state["context"],
                "metadata": final_state["metadata"],
                "memory_context": final_state.get("memory_context", {}),
                "step_results": final_state.get("step_results", {}),
                "step_timings": final_state.get("step_timings", {}),
                "sources": {
                    "documents": final_state["documents"],
                    "kg_results": final_state["kg_results"],
                    "web_results": final_state["web_results"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error running research agent: {e}")
            return {
                "answer": f"I encountered an error while researching your question: {str(e)}",
                "query": query,
                "context": "",
                "metadata": {"error": str(e)},
                "memory_context": {},
                "sources": {"documents": [], "kg_results": [], "web_results": []}
            }
    
    async def _store_chat_interaction(
        self,
        query: str,
        response: str,
        user_id: str,
        profile_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store the chat interaction in history."""
        try:
            # Get or create chat session
            chat_session = await self.memory_manager.get_or_create_session(
                user_id=user_id,
                profile_id=profile_id,
                session_id=session_id
            )
            
            # Generate embeddings for the messages
            query_embedding = await self.embeddings.aembed_query(query)
            response_embedding = await self.embeddings.aembed_query(response)
            
            # Store user query
            await chat_session.add_message(
                role="user",
                content=query,
                embedding=query_embedding,
                metadata={"type": "research_query"}
            )
            
            # Store assistant response
            await chat_session.add_message(
                role="assistant",
                content=response,
                embedding=response_embedding,
                metadata={
                    "type": "research_response",
                    "sources": metadata.get("sources_used", []) if metadata else [],
                    "document_references": metadata.get("document_references", []) if metadata else []
                }
            )
            
            logger.info(f"Stored chat interaction for session {chat_session.session_id}")
            
        except Exception as e:
            logger.error(f"Error storing chat interaction: {e}")
    
    async def close_current_session(
        self, 
        user_id: str, 
        profile_id: str, 
        session_id: Optional[str] = None
    ):
        """Close the current chat session if it exists."""
        try:
            if self.enable_memory:
                # Get the session (don't create if it doesn't exist)
                session_key = session_id or f"{user_id}_{profile_id}_new"
                
                # Check if session exists in memory manager
                if session_key in self.memory_manager._active_sessions:
                    session = self.memory_manager._active_sessions[session_key]
                    await session.close_session()
                    logger.info(f"Closed session {session.session_id} for user {user_id}/{profile_id}")
                else:
                    logger.info(f"No active session found for user {user_id}/{profile_id}")
                    
        except Exception as e:
            logger.error(f"Error closing session: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Close any active sessions when the agent context exits
        if hasattr(self, '_last_user_id') and hasattr(self, '_last_profile_id'):
            await self.close_current_session(
                self._last_user_id, 
                self._last_profile_id, 
                getattr(self, '_last_session_id', None)
            )