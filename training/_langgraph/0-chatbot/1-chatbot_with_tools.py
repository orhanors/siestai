from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, add_messages, START, END
from langchain_community.tools import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from datetime import datetime
# llm = ChatGroq(model="llama-3.1-8b-instant")
llm = ChatOllama(model="qwen3:8b")
memory = MemorySaver()

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

tavily_search_tool = TavilySearchResults(max_results=5)

tools = [tavily_search_tool]

llm_with_tools = llm.bind_tools(tools=tools)

def chatbot(state: ChatState):
    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create system message with current time
    system_message = SystemMessage(content=f"Current time: {current_time}. You are a helpful AI assistant.")
    
    # Combine system message with existing messages
    messages_with_time = [system_message] + state["messages"]
    
    return {
        "messages": [llm_with_tools.invoke(messages_with_time)]
    }

def tools_router(state: ChatState):
    last_message = state["messages"][-1]

    if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    return END

tool_node = ToolNode(tools=tools)

graph = StateGraph(ChatState)

graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("chatbot")
graph.add_conditional_edges("chatbot", tools_router)
graph.add_edge("tool_node", "chatbot")

app = graph.compile(checkpointer=memory)

config = {
    "configuration": {
        "thread_id": 1
    }
}

config = {
    "configurable": {
        "thread_id": i
    }
}
while True:
    user_input = input("User: ")
    if user_input in ["exit", "quit", "end"]:
        break
    result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    print(result)