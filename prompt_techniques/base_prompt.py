from langchain_core.messages import BaseMessage

from models.base_model import BaseLLMModel

class BasePrompt:
    def __init__(self, prompt_instruction_path: str):
        self.prompt_instruction_path = prompt_instruction_path

    def generate_prompt(self, *args):
        pass

    def generate_response(self, model: BaseLLMModel, messages: BaseMessage):
        pass