from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime
from langchain_openai import ChatOpenAI
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion, ReviseAnswer])

# llm = ChatOpenAI(model="gpt-4o")
llm = ChatOllama(model="qwen3:8b")

MAIN_ARCHETYPE = """
You are an expert agent on crypto taxation and portfolio management.
"""

# Actor Agent Prompt
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """ {main_archetype}
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, **list 1-3 search queries separately** for
researching improvements. Do not include them inside the reflection.
"""
       ),
       MessagesPlaceholder(variable_name="messages"),
       ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.now().isoformat(), main_archetype=MAIN_ARCHETYPE)

first_responder_template = actor_prompt_template.partial(first_instruction="Provide a detailed ~250 word answer")
first_responder_chain = first_responder_template | llm.bind_tools(tools = [AnswerQuestion], tool_choice="AnswerQuestion") 

# validator = PydanticToolsParser(tools=[AnswerQuestion])

# response = first_responder_chain.invoke({
#     "messages": [HumanMessage(content="Explain the crypto taxation rules in germany. Make a list of all the rules effecting capital gain and taxation")]
# })


# Revise Answer
revise_instructions = """Revise your previous answer using the new
information.
- You should use the previous critique to add important information to your answer.
    - You MUST include numerical citations in your revised answer to ensure it can be verified.
    - Add a "References" section to the bottom of your answer (which does not Count towards the word limit). In form of:
    - [1] https://example.com
    - [2] https://example.com
- You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revise_prompt_template = actor_prompt_template.partial(first_instruction=revise_instructions)
revise_chain = revise_prompt_template | llm.bind_tools(tools = [ReviseAnswer], tool_choice="ReviseAnswer")

# response = revise_chain.invoke({
#     "messages": [HumanMessage(content="Explain the crypto taxation rules in germany. Make a list of all the rules effecting capital gain and taxation")]
# })

# print(response)