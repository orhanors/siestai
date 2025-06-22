from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class SimpleState(TypedDict):
    count: int

def increment(state: SimpleState) -> SimpleState:
    return {
        "count": state["count"] +1
    }

def should_continue(state: SimpleState) -> str:
    if(state["count"] < 5):
        print(f"Count is {state['count']}")
        return "continue"
    return "stop"

graph = StateGraph(SimpleState)
graph.add_node("increment", increment)
graph.set_entry_point("increment")
graph.add_conditional_edges("increment", should_continue, {
    "continue": "increment",
    "stop": END
})

app = graph.compile()

# print(app.get_graph().draw_mermaid())
# print(app.get_graph().print_ascii())

sample_state = {"count": 0}

response = app.invoke(sample_state)

print(response)