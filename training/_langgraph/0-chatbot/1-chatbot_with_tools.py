from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, add_messages, START, END
from langchain_community.tools import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# llm = ChatGroq(model="llama-3.1-8b-instant")
llm = ChatOllama(model="qwen3:8b")

# Get database configuration from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5439")
DB_NAME = os.getenv("DB_NAME", "postgres")  # Use default postgres database
DB_USER = os.getenv("DB_USER", "siestai_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "siestai_password")

# Construct database URI from environment variables
DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=disable"

print(f"Connecting to database: {DB_HOST}:{DB_PORT}/{DB_NAME}")

try:
    # Use PostgreSQL checkpointer for production
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        print("Successfully connected to PostgreSQL!")
        
        # Setup the checkpointer (CREATE TABLES - only needed first time)
        checkpointer.setup()  # Uncomment this line to create tables
        print("Database tables created successfully!")
        
        class ChatState(TypedDict):
            messages: Annotated[list, add_messages]

        tavily_search_tool = TavilySearchResults(max_results=5)
        tools = [tavily_search_tool]
        llm_with_tools = llm.bind_tools(tools=tools)

        def chatbot(state: ChatState):
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            system_message = SystemMessage(content=f"Current time: {current_time}. You are a helpful AI assistant.")
            messages_with_time = [system_message] + state["messages"]
            return {"messages": [llm_with_tools.invoke(messages_with_time)]}

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
        app = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "1"}}

        print("Chatbot ready! Type 'exit' to quit.")
        while True:
            user_input = input("User: ")
            if user_input in ["exit", "quit", "end"]:
                break
            result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            print(result)

except Exception as e:
    print(f"Database connection failed: {e}")
    print("Falling back to in-memory storage...")
    
    # Fallback to in-memory storage
    memory = MemorySaver()
    
    class ChatState(TypedDict):
        messages: Annotated[list, add_messages]

    tavily_search_tool = TavilySearchResults(max_results=5)
    tools = [tavily_search_tool]
    llm_with_tools = llm.bind_tools(tools=tools)

    def chatbot(state: ChatState):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_message = SystemMessage(content=f"Current time: {current_time}. You are a helpful AI assistant.")
        messages_with_time = [system_message] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages_with_time)]}

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

    config = {"configurable": {"thread_id": "1"}}
    
    print("Chatbot ready (in-memory mode)! Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input in ["exit", "quit", "end"]:
            break
        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        print(result)