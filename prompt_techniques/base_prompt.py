from langchain_core.messages import BaseMessage

from typing import List

from models.base_model import BaseLLMModel

class BasePrompt:
    def __init__(self, prompt_instruction_path: str, table_instruction_path: str = None):
        """Initialize instruction paths"""
        self.prompt_instruction_path = prompt_instruction_path
        self.table_instruction_path = table_instruction_path

    def generate_prompt(self, *args) -> List[BaseMessage]:
        """Generate prompt to extract information not from table"""
        pass

    def generate_table_prompt(self, *args) -> List[BaseMessage]:
        """Generate prompt to extract information from table"""
        pass
    
    def extract_information(self, model: BaseLLMModel, messages: List[BaseMessage]):
        """Extract information not from table"""
        pass
    
    def extract_table_information(self, model: BaseLLMModel, messages: List[BaseMessage]):
        """Extract information from table"""
        pass

    def generate_response(self, model: BaseLLMModel, messages: List[BaseMessage]):
        """Generate response"""
        pass