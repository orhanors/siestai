# GOOGLE_API_KEY and TAVILY_API_KEY must be set in the environment variables
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults
from langchain.agents import initialize_agent, tool
from langchain_ollama import ChatOllama
from langchain.agents import AgentType
from datetime import datetime
from typing import List
from langchain_openai import ChatOpenAI

load_dotenv()

#choose one of the models
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
# llm = ChatOllama(model="llama3.2")
llm = ChatOllama(model="deepseek-r1:8b")
# llm = ChatOpenAI(model="gpt-4o")

search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get the current system time"""
    current_time = datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


tools = [search_tool, get_system_time]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

agent.run("izmirde şuan hava durumu nasıl?")