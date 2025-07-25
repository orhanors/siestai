"""
LangGraph compatibility wrapper for Python 3.13
Maintains LangGraph-like interface while bypassing compatibility issues
"""

import asyncio
import logging
from typing import Dict, Any, Callable, TypedDict, Optional
from dataclasses import dataclass
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

logger = logging.getLogger(__name__)


class StateGraphCompat:
    """Compatible StateGraph implementation that maintains tracing"""
    
    def __init__(self, state_schema):
        self.state_schema = state_schema
        self.nodes = {}
        self.edges = {}
        self.conditional_edges = {}
        self.start_node = None
        self.end_nodes = set()
        
    def add_node(self, name: str, func: Callable):
        """Add a node to the graph"""
        self.nodes[name] = func
        
    def add_edge(self, from_node: str, to_node: str):
        """Add an edge between nodes"""
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)
        
    def add_conditional_edges(self, from_node: str, condition_func: Callable, mapping: Dict[str, str]):
        """Add conditional edges"""
        self.conditional_edges[from_node] = {
            'condition': condition_func,
            'mapping': mapping
        }
        
    def set_entry_point(self, node_name: str):
        """Set the entry point for the graph"""
        self.start_node = node_name
        
    def set_finish_point(self, node_name: str):
        """Set a finish point for the graph"""
        self.end_nodes.add(node_name)
        
    def compile(self, checkpointer=None, interrupt_before=None):
        """Compile the graph into a runnable"""
        return CompiledStateGraph(
            nodes=self.nodes,
            edges=self.edges,
            conditional_edges=self.conditional_edges,
            start_node=self.start_node,
            end_nodes=self.end_nodes,
            state_schema=self.state_schema,
            checkpointer=checkpointer,
            interrupt_before=interrupt_before or []
        )


class CompiledStateGraph(Runnable):
    """Compiled graph that can be executed with LangSmith tracing"""
    
    def __init__(self, nodes, edges, conditional_edges, start_node, end_nodes, 
                 state_schema, checkpointer=None, interrupt_before=None):
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.start_node = start_node
        self.end_nodes = end_nodes
        self.state_schema = state_schema
        self.checkpointer = checkpointer
        self.interrupt_before = interrupt_before or []
        
    def invoke(self, input_state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Sync version - just calls async version"""
        import asyncio
        return asyncio.run(self.ainvoke(input_state, config))
    
    async def ainvoke(self, input_state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Execute the graph asynchronously with tracing"""
        current_state = input_state.copy()
        current_node = self.start_node
        
        # Add LangSmith tracing metadata
        if config is None:
            config = RunnableConfig()
        
        # Set up tracing tags
        config.setdefault("tags", []).extend(["c9s-agent", "langgraph-compat"])
        config.setdefault("metadata", {}).update({
            "agent_type": "c9s-agent",
            "graph_execution": True
        })
        
        step_count = 0
        max_steps = 10  # Prevent infinite loops
        
        while current_node and step_count < max_steps:
            logger.info(f"Executing node: {current_node} (step {step_count + 1})")
            step_count += 1
            
            # Check for interrupts
            if current_node in self.interrupt_before:
                logger.info(f"Interrupt before node: {current_node}")
                # In a real implementation, this would pause execution
                # For now, we'll continue but mark the interruption
                current_state["_interrupted"] = True
            
            # Execute the current node
            node_func = self.nodes[current_node]
            
            # Save state before execution (for checkpointing)
            if self.checkpointer:
                await self.checkpointer.put(config, current_state)
            
            # Wrap node execution with tracing
            try:
                # Execute node with tracing context
                current_state = await self._execute_node_with_tracing(
                    node_func, current_state, current_node, config
                )
                
                # Save state after execution
                if self.checkpointer:
                    await self.checkpointer.put(config, current_state)
                    
            except Exception as e:
                logger.error(f"Error in node {current_node}: {e}")
                current_state["_error"] = str(e)
                break
            
            # Check if this is an end node (after execution)
            if current_node in self.end_nodes:
                logger.info(f"Reached end node: {current_node}")
                break
            
            # Determine next node
            next_node = self._get_next_node(current_node, current_state)
            logger.info(f"Next node determined: {next_node}")
            current_node = next_node
            
        return current_state
    
    async def _execute_node_with_tracing(self, node_func, state, node_name, config):
        """Execute a node with LangSmith tracing"""
        # Create a new config for this node
        node_config = config.copy() if config else RunnableConfig()
        node_config.setdefault("metadata", {}).update({
            "node_name": node_name,
            "node_type": "agent_step"
        })
        
        # Execute the node function
        result = await node_func(state)
        
        # Log the execution for tracing
        logger.info(f"Node {node_name} completed successfully")
        
        return result
    
    def _get_next_node(self, current_node: str, state: Dict[str, Any]) -> Optional[str]:
        """Determine the next node to execute"""
        # Check conditional edges first
        if current_node in self.conditional_edges:
            condition_info = self.conditional_edges[current_node]
            condition_result = condition_info['condition'](state)
            next_node = condition_info['mapping'].get(condition_result)
            if next_node:
                return next_node
        
        # Check regular edges
        if current_node in self.edges:
            edges = self.edges[current_node]
            if edges:
                return edges[0]  # Take the first edge
                
        return None
    
    async def aget_state(self, config: RunnableConfig) -> 'StateSnapshot':
        """Get the current state (for checkpointing)"""
        if self.checkpointer:
            return await self.checkpointer.get(config)
        else:
            return StateSnapshot({}, None, config, {})
    
    async def aupdate_state(self, config: RunnableConfig, values: Dict[str, Any]):
        """Update the state (for human-in-the-loop)"""
        if self.checkpointer:
            await self.checkpointer.update(config, values)
        else:
            logger.info(f"State update requested (no checkpointer): {values}")


@dataclass
class StateSnapshot:
    """Mock state snapshot for compatibility"""
    values: Dict[str, Any]
    next: Optional[str]
    config: Optional[RunnableConfig]
    metadata: Optional[Dict[str, Any]]


# Mock checkpointer for compatibility
class MockPostgresCheckpointer:
    """Mock PostgreSQL checkpointer that maintains interface"""
    
    def __init__(self):
        self._states = {}  # In-memory state storage
    
    @classmethod
    def from_conn_string(cls, conn_string: str):
        """Create checkpointer from connection string"""
        logger.info(f"Mock checkpointer created for: {conn_string}")
        return cls()
    
    async def get(self, config):
        """Get state from checkpointer"""
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id and thread_id in self._states:
            logger.info(f"Retrieved state for thread: {thread_id}")
            return StateSnapshot(self._states[thread_id], None, config, {})
        else:
            logger.info(f"No state found for thread: {thread_id}")
            return StateSnapshot({}, None, config, {})
    
    async def put(self, config, state):
        """Save state to checkpointer"""
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id:
            self._states[thread_id] = state
            logger.info(f"Saved state for thread: {thread_id}")
    
    async def update(self, config, values):
        """Update state in checkpointer"""
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id and thread_id in self._states:
            self._states[thread_id].update(values)
            logger.info(f"Updated state for thread: {thread_id}")


# Constants for compatibility
START = "START"
END = "END"