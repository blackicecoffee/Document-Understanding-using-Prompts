from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from typing import List
import re

from prompt_techniques.base_prompt import BasePrompt
from models.base_model import BaseLLMModel

"""
Vanilla prompting technique
"""

class VanillaPrompt(BasePrompt):
    def __init__(self, prompt_instruction_path):
        super().__init__(prompt_instruction_path)

    def generate_prompt(self, fields: List[str], image_data: str):
        with open(self.prompt_instruction_path, "r") as f:
            prompt = f.read()
        
        prompt = prompt.format(fields=fields)

        contents = [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data,
                    "detail": "high"
                }
            }
        ]

        messages = [
            SystemMessage(content="You are an helpful AI assistant that help user extract data from document image to JSON object."),
            HumanMessage(content=contents)
        ]

        return messages

    
    async def generate_response(self, model: BaseLLMModel, fields: dict, image_data: str):
        prompt_messages = self.generate_prompt(fields=fields, image_data=image_data)

        response = await model.generate(prompt_messages)

        results = response.replace("json", "").replace("\n", "").replace("```", "").replace("'", '"')
        results = re.search(r'\{.*?\}', results).group()

        return results