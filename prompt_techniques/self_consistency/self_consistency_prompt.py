from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from collections import Counter
from typing import List
import asyncio
import json
import re

from models.base_model import BaseLLMModel
from prompt_techniques.base_prompt import BasePrompt
from prompt_techniques.vanilla_prompt.vanilla_prompt import VanillaPrompt
from prompt_techniques.Few_Shots.few_shot_prompt import FewShotsPrompt
from helpers.merge_results import merge

"""
Self-consistency prompting technique
"""

class SelfConsistencyPrompt(BasePrompt):
    def __init__(
            self, 
            prompt_instruction_path: str, 
            table_instruction_path: str, 
            baseline_prompt: str,
            examples: List[dict] = None,
            k: int = 3
        ):
        super().__init__(prompt_instruction_path, table_instruction_path=table_instruction_path)
        self.baseline_prompt = baseline_prompt
        self.examples = examples
        self.k = k

    def generate_prompt(self, *args):
        return super().generate_prompt(*args)
    
    def generate_table_prompt(self, *args):
        return super().generate_table_prompt(*args)

    async def extract_information(self, model: BaseLLMModel, fields: dict, image_data: str):
        if self.baseline_prompt == "vanilla":
            tasks = [
                VanillaPrompt(
                    prompt_instruction_path=self.prompt_instruction_path
                ).extract_information(
                    model=model,
                    fields=fields,
                    image_data=image_data
                ) for _ in range(self.k)
            ]

            resutls = await asyncio.gather(*tasks)
        
        elif self.baseline_prompt == "few_shots":
            tasks = [
                FewShotsPrompt(
                    prompt_instruction_path=self.prompt_instruction_path,
                    table_instruction_path=self.table_instruction_path,
                    examples=self.examples
                ).extract_information(
                    model=model,
                    fields=fields,
                    image_data=image_data
                ) for _ in range(self.k)
            ]
            resutls = await asyncio.gather(*tasks)

        # Aggregate results
        final_results = {}

        for field in fields:
            value_lst = [res[field] for res in resutls if field in res]

            counter = Counter(value_lst)
            try:
                most_common_value, _ = counter.most_common(1)[0]
            except Exception:
                most_common_value = ""
            
            final_results[field] = most_common_value
            
        return final_results

    async def extract_table_information(self, model: BaseLLMModel, table_columns: List[dict], image_data:str):
        if self.baseline_prompt == "vanilla":
            tasks = [
                VanillaPrompt(
                    prompt_instruction_path=self.prompt_instruction_path,
                    table_instruction_path=self.table_instruction_path
                ).extract_table_information(
                    model=model,
                    table_columns=table_columns,
                    image_data=image_data
                ) for _ in range(self.k)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        elif self.baseline_prompt == "few_shots":
            tasks = [
                FewShotsPrompt(
                    prompt_instruction_path=self.prompt_instruction_path,
                    table_instruction_path=self.table_instruction_path,
                    examples=self.examples
                ).extract_table_information(
                    model=model,
                    table_columns=table_columns,
                    image_data=image_data
                ) for _ in range(self.k)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        final_results = []
    
        counter = Counter([len(res) for res in results])
        most_common_table_length, _ = counter.most_common(1)[0]

        table_lst = [res for res in results if len(res) == most_common_table_length]

        for row_idx in range(most_common_table_length):
            final_row = {}

            for col in table_columns[0].keys():
                try:
                    col_value_lst = [table[row_idx][col] for table in table_lst if col in table[row_idx]]
                except Exception:
                    col_value_lst = []
                
                if len(col_value_lst) == 0:
                    final_row[col] = ""
                    continue

                col_counter = Counter(col_value_lst)
                try:
                    most_common_col_value, _ = col_counter.most_common(1)[0]
                except Exception:
                    most_common_col_value = ""
                
                final_row[col] = most_common_col_value
            
            if all(len(val) == 0 for val in final_row.values()): continue

            final_results.append(final_row)

        return final_results

    async def generate_response(self, model: BaseLLMModel, fields: dict, table_columns: List[dict], image_data: str):
        results = None
        table_results = None

        async with asyncio.TaskGroup() as tg:
            information_task = tg.create_task(self.extract_information(
                model=model,
                fields=fields,
                image_data=image_data
            ))

            if self.table_instruction_path and table_columns != None:
                table_task = tg.create_task(self.extract_table_information(
                    model=model,
                    table_columns=table_columns,
                    image_data=image_data
                ))

        results = information_task.result()

        if self.table_instruction_path and table_columns != None:
            table_results = table_task.result()

        final_result = merge(information=results, table=table_results)

        return final_result
    