from langchain_together.chat_models import ChatTogether
from langchain_core.messages import BaseMessage

import os
from typing import Optional, List

from models.base_model import BaseLLMModel

class LlamaVision(BaseLLMModel):
    def __init__(
            self, 
            temperature: Optional[float] = 0, 
            max_tokens: Optional[int] = None, 
            max_retries: Optional[int] = 2, 
            timeout: Optional[float] = None, 
            logprobs: Optional[bool] = False
        ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.logprobs = logprobs
        self.model_name = "Llama-3.2-11B-Vision-Instruct-Turbo"
        
        self.model = ChatTogether(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout=timeout,
            logprobs=logprobs,
            api_key=os.getenv("TOGETHER_API_KEY")
        )

    async def generate(self, messages: List[BaseMessage]):
        results = await self.model.ainvoke(messages)

        return results.content