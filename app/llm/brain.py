from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from enum import Enum
from datetime import datetime
import json

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_AGENT_INFO_PATH = "app/agent-info.json"

class LLM_MODEL(Enum):
    # API Models
    OPENAI_GPT_4O = "gpt-4o"
    GROQ_LLAMA3_1_8B_INSTANT = "llama-3.1-8b-instant" #free-tier option
    
    # Local Models
    LLAMA3_2 = "llama3.2:latest"
    DEEPSEEK_R1_8B = "deepseek-r1:8b"
    QWEN3_8B = "qwen3:8b"

    @classmethod
    def is_local_model(cls, value):
        local_models = [cls.LLAMA3_2, cls.DEEPSEEK_R1_8B, cls.QWEN3_8B]
        return value in local_models


class Brain:
    def __init__(self, model_name: LLM_MODEL, agent_name: str):
        self.agent_name = agent_name  # Store the agent name
        if not self.is_valid_agent(agent_name=agent_name):
            raise ValueError(f"Agent {agent_name} not found in agent-info.json, Valid agents are: {self.get_valid_agents()}")

        match model_name:
            case LLM_MODEL.OPENAI_GPT_4O:
                self.model = ChatOpenAI(model=model_name.value)

            case LLM_MODEL.GROQ_LLAMA3_1_8B_INSTANT:
                self.model = ChatGroq(model=model_name.value)
                
            case _ if LLM_MODEL.is_local_model(model_name):
                self.model = OllamaLLM(model=model_name.value)
            case _:
                raise ValueError(f"Model {model_name} not supported")

        prompt = ChatPromptTemplate.from_template(self.generate_system_prompt()).partial(current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.chain = prompt | self.model
        logger.info(f"🧠 Brain initialized!")
        logger.info(f"🤖 LLM Model: {model_name.value}")
        logger.info(f"🤖 Agent Name: {agent_name}")
        logger.info(f"📝 System Template: \n {self.generate_system_prompt()}")
        
    def chat(self, profile_id, message):
        return self.chain.invoke({"profile_id": profile_id, "question": message})
    
    def add_tools(self, tools: list):
        # Create a new chain that includes tools
        prompt = ChatPromptTemplate.from_template(self.generate_system_prompt()).partial(current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.chain = prompt | self.model.bind_tools(tools=tools)

    def is_valid_agent(self, agent_name: str) -> bool:
        with open(BASE_AGENT_INFO_PATH, "r") as f:
            agent_info = json.load(f)
        return agent_name in agent_info.keys()
    
    def get_valid_agents(self) -> list:
        with open(BASE_AGENT_INFO_PATH, "r") as f:
            agent_info = json.load(f)
        return list(agent_info.keys())

    def generate_system_prompt(self) -> str:
        with open(BASE_AGENT_INFO_PATH, "r") as f:
            agent_info = json.load(f)
        
        agent = agent_info[self.agent_name]
        name = agent["name"]
        purpose = agent["purpose"]
        main_context = agent["main_context"]
        parameters = agent["parameters"]

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_prompt = f"Current time: {current_time}\n"
        # TODO: Raise errors for some fields that are not provided
        if name:
            system_prompt += f"Your name is: {name}\n"
        if purpose:
            system_prompt += f"Your purpose is: {purpose}\n"
        if main_context:
            system_prompt += f"Your main context is: {main_context}\n"
        if parameters:
            #TODO: Change base parameters prompt
            system_prompt += "You are provided following parameters if needed for tool calling:\n"
            for param_name, parameter in parameters.items():
                system_prompt += f"{parameter['prefix']}: {"{" + param_name + "}"}\n"
        return system_prompt
    