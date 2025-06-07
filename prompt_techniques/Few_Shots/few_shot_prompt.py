from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from typing import List
import asyncio
import json
import re

from prompt_techniques.base_prompt import BasePrompt
from models.base_model import BaseLLMModel
from helpers.merge_results import merge

"""
Few Shot Prompting Techinique
"""

class FewShotsPrompt(BasePrompt):
    def __init__(self, prompt_instruction_path: str, table_instruction_path: str, examples: List[dict]):
        super().__init__(prompt_instruction_path, table_instruction_path)
        self.examples = examples

    def generate_prompt(self, fields: List[str], image_data: str) -> List[BaseMessage]:
        with open(self.prompt_instruction_path, "r") as f:
            prompt = f.read()

        samples = []

        for example in self.examples:
            sample_prompt = prompt.format(fields=example["fields"])

            sample_content = [
                {
                    "type": "text",
                    "text": sample_prompt
                },
                {
                    "type": "text",
                    "text": "Document image:\n"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": example["image_data"],
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": "# Output:\n"
                },
                {
                    "type": "text",
                    "text": f"{example["formal"]}"
                }
            ]

            samples.append(HumanMessage(content=sample_content))

        prompt = prompt.format(fields=fields)

        contents = [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "text",
                "text": "Document image:\n"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data,
                    "detail": "high"
                }
            },
            {
                "type": "text",
                "text": "# Output:\n"
            }
        ]

        messages = [
            SystemMessage(content="You are an helpful AI assistant that help user extract data from document image to JSON object. You will be given some demonstrations about your tasks. Based on that, complete the given tasks accurately."),
        ]

        messages += samples
        messages += [HumanMessage(content=contents)]

        return messages

    def generate_table_prompt(self, table_columns: List[dict], image_data: str) -> List[BaseMessage]:
        with open(self.table_instruction_path, "r") as f:
            prompt = f.read()

        samples = []

        for example in self.examples:
            sample_prompt = prompt.format(table_columns=example["table_columns"])

            sample_content = [
                {
                    "type": "text",
                    "text": sample_prompt
                },
                {
                    "type": "text",
                    "text": "Document image:\n"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": example["image_data"],
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": "# Output:\n"
                },
                {
                    "type": "text",
                    "text": f"{example["table"]}"
                }
            ]

            samples.append(HumanMessage(content=sample_content))


        prompt = prompt.format(table_columns=table_columns)

        contents = [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "text",
                "text": "Document image:\n"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data,
                    "detail": "high"
                }
            },
            {
                "type": "text",
                "text": "# Output:\n"
            }
        ]

        messages = [
            SystemMessage(content="You are an helpful AI assistant that help user extract table data from document image to JSON array. You will be given some demonstrations about your tasks. Based on that, complete the given tasks accurately."),
        ]

        messages += samples
        messages += [HumanMessage(content=contents)]

        return messages
    
    async def extract_information(self, model: BaseLLMModel, fields: dict, image_data: str):
        await asyncio.sleep(0)
        
        prompt_messages = self.generate_prompt(fields=fields, image_data=image_data)

        response = await model.generate(prompt_messages)

        results = response.replace("json", "").replace("\n", "").replace("```", "").replace("'", '"')

        results = re.search(r'\{.*?\}', results).group()

        results_json = json.loads(results)

        return results_json
    
    async def extract_table_information(self, model: BaseLLMModel, table_columns: List[dict], image_data: str):
        await asyncio.sleep(0)

        prompt_messages = self.generate_table_prompt(table_columns=table_columns, image_data=image_data)

        response = await model.generate(prompt_messages)

        results = response.replace("json", "").replace("\n", "").replace("```", "").replace("'", '"')

        try:
            results = re.search(r'\[.*?\]', results).group()
            
            results_json = json.loads(results)
        except:
            results_json = []

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

        final_result = merge(information=results, table=table_results)

        return final_result