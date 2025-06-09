from langchain_together.chat_models import ChatTogether
from langchain_core.messages import BaseMessage

import os
from typing import Optional, List

from models.base_model import BaseLLMModel

class QwenVision(BaseLLMModel):
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
        self.model_name = "Qwen2-VL-72B-Instruct"
        
        self.model = ChatTogether(
            model="Qwen/Qwen2-VL-72B-Instruct",
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