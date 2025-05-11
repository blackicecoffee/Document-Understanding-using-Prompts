from prompt_techniques.base_prompt import BasePrompt

"""
Self-consistency prompting technique
"""

class SelfConsistencyPrompt(BasePrompt):
    def __init__(self, prompt_instruction_path):
        super().__init__(prompt_instruction_path)

    def generate_prompt(self, *args):
        return super().generate_prompt(*args)
    
    def generate_response(self, model, messages):
        return super().generate_response(model, messages)