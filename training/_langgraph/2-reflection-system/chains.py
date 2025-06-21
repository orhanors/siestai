from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a twitter techie influencer assistan tasked with writing excellent twitter posts"),
    MessagesPlaceholder(variable_name="messages")
])

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the tweet."),
    MessagesPlaceholder(variable_name="messages")
])

# llm = ChatOpenAI(model="gpt-4o")
# llm = ChatOllama(model="deepseek-r1:8b")
llm = ChatOllama(model="llama3.2")

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm