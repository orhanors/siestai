from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
import operator
class SimpleState(TypedDict):
    count: int
    # Annotated with operator will help us to update the state properly with previous state
    sum: Annotated[int, operator.add]
    history: Annotated[List[int], operator.concat]

def increment(state: SimpleState) -> SimpleState:
    new_count = state["count"] + 1
    return {
        "count": new_count,
        "sum": new_count,
        "history": [new_count]
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