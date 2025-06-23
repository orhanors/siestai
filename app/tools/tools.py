from langchain_community.tools import TavilySearchResults


class BrainTools:
    def __init__(self):
        self.tools = []

    def get_tools(self):
        return self.tools
    
    def add_search_tool(self):
        self.tools.append(TavilySearchResults(max_results=5))
        return self
    