# GOOGLE_API_KEY and TAVILY_API_KEY must be set in the environment variables
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults
from langchain.agents import initialize_agent, tool
from langchain.agents import AgentType
from datetime import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

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

agent.run("What is the exact date of the geoffrey hinton's birthday?")