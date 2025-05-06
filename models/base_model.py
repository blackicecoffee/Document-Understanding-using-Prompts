from langchain_core.messages import BaseMessage

class BaseLLMModel:
    def __init__(self, *args):
        pass

    def generate(self, prompt: BaseMessage):
        pass