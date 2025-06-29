from typing import TypedDict
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from llm.brain import LLM_MODEL
from rawdata.intercom import IntercomDataProvider

#CHOOSE THE BASE MODEL
LLM_MODEL = LLM_MODEL.GROQ_LLAMA3_1_8B_INSTANT
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

intercom_data_provider = IntercomDataProvider()
articles, next_page = intercom_data_provider.get_articles(per_page=10, page=33)
print("next_page::", next_page)