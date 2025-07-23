"""Research Agent with LangGraph for document retrieval and web search."""

import os
import logging
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import SystemMessage, HumanMessage

from app.memory.database.database import vector_search, hybrid_search
from app.memory.knowledge_graph.knowledge_graph import KGClient
from app.services.embedding_service import get_embeddings

logger = logging.getLogger(__name__)


class ResearchState(TypedDict):
    """State for the research agent."""
    query: str
    documents: List[Dict[str, Any]]
    kg_results: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    final_answer: str
    context: str
    metadata: Dict[str, Any]


class ResearchAgent:
    """LangGraph-based research agent with document retrieval and web search."""
    
    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_documents: int = 5,
        max_web_results: int = 3,
        enable_kg: bool = True,
        enable_web_search: bool = True
    ):
        """Initialize the research agent.
        
        Args:
            model: OpenAI model to use
            temperature: Model temperature
            max_documents: Maximum documents to retrieve
            max_web_results: Maximum web search results
            enable_kg: Whether to use knowledge graph
            enable_web_search: Whether to use web search
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.max_documents = max_documents
        self.max_web_results = max_web_results
        self.enable_kg = enable_kg
        self.enable_web_search = enable_web_search
        
        # Initialize services
        self.embeddings = get_embeddings()
        
        if self.enable_kg:
            self.kg_client = KGClient()
            
        if self.enable_web_search:
            self.web_search = TavilySearchResults(
                max_results=max_web_results,
                search_depth="advanced"
            )
        
        # Build the graph
        self.graph = self._build_graph()
    
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
            
            # Create the prompt
            system_message = SystemMessage(content="""You are a research assistant that synthesizes information from multiple sources to provide comprehensive, accurate answers.

Instructions:
1. Use the provided context from documents, knowledge graph, and web search results
2. Provide a clear, well-structured answer to the user's question
3. Cite your sources when possible
4. If information is conflicting, acknowledge this and explain
5. If you don't have enough information, say so clearly
6. Be concise but thorough
7. When referencing documents from the knowledge base, you can refer to them by their titles""")
            
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
        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("search_kg", self.search_knowledge_graph)
        workflow.add_node("web_search", self.web_search_step)
        workflow.add_node("synthesize", self.synthesize_answer)
        
        # Add edges
        workflow.add_edge(START, "retrieve_documents")
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
    
    async def research(self, query: str) -> Dict[str, Any]:
        """Run the research agent on a query.
        
        Args:
            query: The research query
            
        Returns:
            Dictionary with the answer and metadata
        """
        initial_state = ResearchState(
            query=query,
            documents=[],
            kg_results=[],
            web_results=[],
            final_answer="",
            context="",
            metadata={}
        )
        
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                "answer": final_state["final_answer"],
                "query": final_state["query"],
                "context": final_state["context"],
                "metadata": final_state["metadata"],
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
                "sources": {"documents": [], "kg_results": [], "web_results": []}
            }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass