from typing import TypedDict
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from llm.brain import Brain, LLM_MODEL
from tools.tools import BrainTools
from langgraph.graph import StateGraph, START, END

#CHOOSE THE BASE MODEL
LLM_MODEL = LLM_MODEL.GROQ_LLAMA3_1_8B_INSTANT
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

#Define the tools
tools = BrainTools()
tools.add_search_tool()

#CREATE THE BRAIN with TOOLS
brain = Brain(LLM_MODEL, "generic_agent")
brain.add_tools(tools=tools.get_tools())


graph = StateGraph(AgentState)