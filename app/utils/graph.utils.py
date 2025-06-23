from IPython.display import Image, display
from langgraph.graph import CompiledGraph

class GraphUtils:
    @staticmethod
    def print_graph_image(graph: CompiledGraph):
        """
        Print the graph as an image using mermaid PNG generation.
        
        Args:
            graph: A LangGraph compiled graph object
        """
        try:
            display(Image(graph.get_graph().draw_mermaid_png()))
        except Exception as e:
            print(f"Could not display graph image: {e}")
            print("This requires some extra dependencies and is optional")
            # Fallback to text representation
            print("Mermaid Diagram:")
            print(graph.get_graph().draw_mermaid())
            print("\nASCII Diagram:")
            print(graph.get_graph().print_ascii())