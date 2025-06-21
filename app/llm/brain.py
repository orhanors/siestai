from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from enum import Enum

class LLM_MODEL(Enum):
    LLAMA3_2 = "llama3.2:latest"
    DEEPSEEK_R1_8B = "deepseek-r1:8b"

class Brain:
    TEMPLATE = """
    You are an expert in the field of Crypto trading and its taxation.
    You can analyze users transactions and answer questions based on their profile_id.

    Here is the user's profile_id: {profile_id}

    Here is the context: {context}

    Here is the question: {question}
    """
    def __init__(self, model_name: str = "llama3.2"):
        self.model = OllamaLLM(model=model_name)
        prompt = ChatPromptTemplate.from_template(self.TEMPLATE)
        self.chain = prompt | self.model
        
    def chat(self, profile_id, context, message):
        return self.chain.invoke({"profile_id": profile_id, "context": context, "question": message})
    
    