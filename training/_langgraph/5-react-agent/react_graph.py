from dotenv import load_dotenv

load_dotenv()

from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import StateGraph, START, END
from nodes import reason_node, act_node
from react_state import AgentState

REASON_NODE = "reason_node"
ACT_NODE = "act_node"

def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    return ACT_NODE

graph = StateGraph(AgentState)

graph.add_node(REASON_NODE, reason_node)
graph.set_entry_point(REASON_NODE)
graph.add_node(ACT_NODE, act_node)

graph.add_conditional_edges(REASON_NODE, should_continue, {
    ACT_NODE: ACT_NODE,
    END: END
})

graph.add_edge(ACT_NODE, REASON_NODE)

app = graph.compile()

print(app.get_graph().draw_mermaid())
print(app.get_graph().print_ascii())

response = app.invoke({"input": "Bu yılın aralık ayında izmirden taylanda gitmek için en uygun uçak bileti fiyatı ne kadar olur?"})
print(response)