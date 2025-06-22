from typing import List

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import START, END, MessageGraph

from chains import first_responder_chain, revise_chain
from execute_tools import execute_tools

MAX_ITERATIONS = 2

graph = MessageGraph()

graph.add_node("draft", first_responder_chain)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revise_chain)

graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "revisor")

def evenp_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)

    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

graph.add_conditional_edges("revisor", evenp_loop)
graph.set_entry_point("draft")

app = graph.compile()

print(app.get_graph().draw_mermaid())
print(app.get_graph().print_ascii())

response = app.invoke("I want to buy some popular altcoins. suggest me 3 altcoins to buy with current prices and info")
print(response[-1].tool_calls[0]["args"])