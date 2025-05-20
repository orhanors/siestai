from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.2")

template = """"
You are an expert in the field of Crypto trading and its taxation.

Here are some information for you: {context}

Here is the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

print("COMES JEER")
result = chain.invoke({"context": """
Portugal Requirements
The capital gain method is FIFO applied to each connection separately. If the crypto has been held for 365 days or more the sale is tax-free. If the crypto has been held for less  than 365 days the sale is taxed. Trades from crypto to crypto are tax-free (the charging price of the crypto sent is given to the one received).If the counterpart of an outgoing transaction or a trade is not resident in the EU, SEE, or a country with a Double Tax Agreement signed with Portugal points 2 and 4 are not applied and that transaction is taxed even if the asset was bought in previous 364 days or was crypto to crypto
 """, 
 "question": "How to calculate taxes on crypto trading in Portugal?"})

print(result)