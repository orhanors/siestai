import json
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_community.tools import TavilySearchResults

#Create the Tavily search tool
tavily_search_tool = TavilySearchResults(max_results=5)

#Function to execute search queries from AnswerQuestion tool call
def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    last_ai_message: AIMessage = state[-1]

    #Extract tool calls from the last AI message
    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return []
    
    #Process the AnswerQuestion or ReviseAnswer tool calls to extract search queries
    tool_messages = []
    for tool_call in last_ai_message.tool_calls:
        if(tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]):
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries", [])

            #Execute each search query using the Tavily search tool
            query_results = {}
            for query in search_queries:
                result = tavily_search_tool.invoke(query)
                query_results[query] = result

            #Create a new tool message with the search results
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(query_results),
                    name="TavilySearchResults",
                    tool_call_id=call_id
                )
            )
    return tool_messages

test_state = [
    HumanMessage(content="What is the capital of France?"),
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "AnswerQuestion",
                "args": {
                    "answer": "The capital of France is Paris.",
                    "search_queries": ["capital of France", "population of Paris"],
                    "reflection": {
                        "missing": "The answer should include the country name.",
                        "superfluous": "The answer should not include the population of Paris."
                    }
                },
                "id": "123"
            }
        ]
    )
]

# print(execute_tools(test_state))