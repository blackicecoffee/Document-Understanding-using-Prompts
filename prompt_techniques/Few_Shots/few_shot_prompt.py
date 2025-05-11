from prompt_techniques.base_prompt import BasePrompt

"""
Few Shot Prompting Techinique
"""

class FewShotsPrompt(BasePrompt):
    def __init__(self, prompt_instruction_path):
        super().__init__(prompt_instruction_path)

    def generate_prompt(self, *args):
        return super().generate_prompt(*args)
    
    def generate_response(self, model, messages):
        return super().generate_response(model, messages)