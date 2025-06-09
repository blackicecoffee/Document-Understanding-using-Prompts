from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from typing import List
import asyncio
import json
import re

from prompt_techniques.base_prompt import BasePrompt
from models.base_model import BaseLLMModel
from helpers.merge_results import merge
from helpers.string_handler import fix_unescaped_inner_quotes

"""
Vanilla prompting technique
"""

class VanillaPrompt(BasePrompt):
    def __init__(self, prompt_instruction_path: str, table_instruction_path: str = None):
        super().__init__(prompt_instruction_path, table_instruction_path)

    def generate_prompt(self, fields: List[str], image_data: str) -> List[BaseMessage]:
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

    def generate_table_prompt(self, table_columns: List[dict], image_data: str) -> List[BaseMessage]:
        with open(self.table_instruction_path, "r") as f:
            prompt = f.read()

        prompt = prompt.format(table_columns=table_columns)

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
            SystemMessage(content="You are an helpful AI assistant that help user extract table data from document image to JSON array."),
            HumanMessage(content=contents)
        ]

        return messages

    async def extract_information(self, model: BaseLLMModel, fields: dict, image_data: str):
        await asyncio.sleep(0)
        
        prompt_messages = self.generate_prompt(fields=fields, image_data=image_data)

        response = await model.generate(prompt_messages)

        results = response.replace("json", "").replace("\n", "").replace("```", "")
       
        try:
            results = re.search(r'\{.*?\}', results).group()
            results = fix_unescaped_inner_quotes(results)

            results_json = json.loads(results)
        except Exception:
            results_json = fields
        return results_json
    
    async def extract_table_information(self, model: BaseLLMModel, table_columns: List[dict], image_data: str):
        await asyncio.sleep(0)

        prompt_messages = self.generate_table_prompt(table_columns=table_columns, image_data=image_data)

        response = await model.generate(prompt_messages)

        results = f"""{response}""".replace("json", "").replace("\n", "").replace("```", "")

        try:
            results = re.search(r'\[.*?\]', results).group()
            results = fix_unescaped_inner_quotes(results)
            results_json = json.loads(results)
        except:
            results_json = table_columns

        return results_json
    
    async def generate_response(self, model: BaseLLMModel, fields: dict, table_columns: List[dict], image_data: str):
        results = None
        table_results = None

        async with asyncio.TaskGroup() as tg:
            information_task = tg.create_task(self.extract_information(
                model=model,
                fields=fields,
                image_data=image_data
            ))

            if self.table_instruction_path:
                table_task = tg.create_task(self.extract_table_information(
                    model=model,
                    table_columns=table_columns,
                    image_data=image_data
                ))

        results = information_task.result()

        if self.table_instruction_path:
            table_results = table_task.result()
        try:
            final_result = merge(information=results, table=table_results)
        except Exception:
            print(f"\nFormal: {results}\n")
            print(f"Table: {table_results}\n\n")
        return final_result