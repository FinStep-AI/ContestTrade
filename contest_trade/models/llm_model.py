"""
OpenAI Model Implementation

This module provides an implementation of the BaseAgentModel for OpenAI models.
It follows an async-first approach, where the primary implementation is the
asynchronous streaming method (a_stream_run).
"""
import os
import sys
import httpx
import openai
import asyncio
from pathlib import Path
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from typing import Dict, List, Optional, AsyncIterator, Callable
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from config.config import cfg

from models.base_agent_model import (
    BaseAgentModel,
    AsyncResponseStream,
    StreamingChunk,
    ModelResponse
)

class LLMModelConfig:
    def __init__(self, model_name: str, api_key: str, base_url: str,
                 max_retries: int = 3, retry_delay: float = 20.0, timeout: float = 60.0, extra_headers: dict = None, proxys: dict = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.extra_headers = extra_headers
        self.proxys = proxys


class LLMModel(BaseAgentModel):
    """
    OpenAI model implementation.
    
    This class provides a concrete implementation of the BaseAgentModel.
    Following the async-first approach, it only implements the a_stream_run method,
    while all other methods (run, a_run, stream_run) are handled by the base class.
    """
    
    def __init__(
        self,
        config: LLMModelConfig,
        **kwargs
    ):
        """
        Initialize the OpenAI model.
        
        Args:
            model_name: The name of the OpenAI model to use
            api_key: OpenAI API key (optional)
            base_url: Base URL for API requests (optional)
            **kwargs: Additional configuration parameters
        """
        super().__init__(config, **kwargs)
        
        self.config = config
        self.model_name = config.model_name
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.extra_headers = config.extra_headers
        self.proxys = config.proxys
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.base_url is None:
            self.base_url = os.environ.get("OPENAI_BASE_URL")

        # Initialize synchronous client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=httpx.Client(proxy=self.proxys) if self.proxys else None
        )
        
        # Initialize asynchronous client
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=httpx.AsyncClient(proxy=self.proxys) if self.proxys else None
        )

        if self.extra_headers is not None:
            self.client = self.client.with_options(
                default_headers=self.extra_headers)
            self.async_client = self.async_client.with_options(
                default_headers=self.extra_headers)
    
    def _process_chunk(self, chunk: ChatCompletionChunk) -> StreamingChunk[str]:
        """Process a streaming chunk from OpenAI."""
        # Extract content from the chunk
        #print(chunk.choices)
        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
            content = chunk.choices[0].delta.reasoning_content
            is_reasoning = True
        else:
            content = chunk.choices[0].delta.content or ""
            is_reasoning = False
        is_finished = len(chunk.choices) > 0 and chunk.choices[0].finish_reason is not None
        
        return StreamingChunk(
            content=content,
            is_finished=is_finished,
            raw_chunk=chunk,
            is_reasoning=is_reasoning
        )

    async def a_run_with_semaphore(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        timeout: Optional[float] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        **kwargs
    ) -> ModelResponse[str]:
        """
        Run the model asynchronously and stream the response with retry mechanism.
        
        This is the primary implementation that all other methods
        (run, a_run, stream_run) will use as their base.
        """
        async with semaphore:
            try:
                response = await self.a_run(
                    messages, 
                    temperature=temperature, 
                    max_tokens=max_tokens, 
                    max_retries=max_retries, 
                    retry_delay=retry_delay, 
                    timeout=timeout, 
                    **kwargs
                )
                return response
            except Exception as e:
                print(f"Error: {e}")
                return None


    async def a_run(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        verbose: bool = False,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        timeout: Optional[float] = None,
        post_process_func: Optional[Callable[[str], str]] = None,
        **kwargs
    ) -> ModelResponse[str]:
        """
        Run the model asynchronously and return a complete response.
        
        This is a wrapper around a_stream_run() that collects all chunks
        and combines them into a single response. Subclasses should
        implement a_stream_run() as the primary method and this method
        will be handled automatically.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            A ModelResponse containing the generated content
        """

        if max_retries is None:
            max_retries = getattr(self, 'config', LLMModelConfig("", "", "")).max_retries
        if retry_delay is None:
            retry_delay = getattr(self, 'config', LLMModelConfig("", "", "")).retry_delay
        if timeout is None:
            timeout = 60

        for attempt in range(max_retries + 1):
            try:
                # Get the stream
                stream = await self.a_stream_run(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    **kwargs
                )
                
                # Collect all chunks
                reasoning_content = ""
                full_content = ""
                raw_chunks = []
                
                async for chunk in stream:
                    if chunk.is_reasoning:
                        reasoning_content += chunk.content
                    else:
                        full_content += chunk.content
                    if chunk.raw_chunk is not None:
                        raw_chunks.append(chunk.raw_chunk)
                        if verbose:
                            print(chunk.content, end="", flush=True)
            
                if post_process_func is not None:
                    proc_response = post_process_func(full_content)
                else:
                    proc_response = None

                # Create a response with the collected content
                return ModelResponse(
                    content=self.postprocess_response(full_content),
                    reasoning_content=reasoning_content,
                    model_name=self.model_name,
                    raw_response=raw_chunks if raw_chunks else None,
                    proc_response=proc_response
                )
            except Exception as e:
                if attempt < max_retries:
                    print(f"🔄 LLM API调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {type(e).__name__}: {e}")
                    print(f"⏳ 等待 {retry_delay} 秒后重试...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    print(f"❌ LLM API调用最终失败，已重试 {max_retries} 次: {type(e).__name__}: {e}")
                    raise
    

    async def a_stream_run(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> AsyncResponseStream[str]:
        """
        Run the model asynchronously and stream the response with retry mechanism.
        
        This is the primary implementation that all other methods
        (run, a_run, stream_run) will use as their base.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay between retries in seconds (default: 20.0)
            timeout: Timeout for each attempt in seconds (default: 60.0)
            **kwargs: Additional model-specific parameters
            
        Returns:
            An AsyncResponseStream that yields chunks of the generated content
        """
        
        # 使用配置中的默认值（如果未指定）
        if max_retries is None:
            max_retries = getattr(self, 'config', LLMModelConfig("", "", "")).max_retries
        if retry_delay is None:
            retry_delay = getattr(self, 'config', LLMModelConfig("", "", "")).retry_delay
        if timeout is None:
            timeout = getattr(self, 'config', LLMModelConfig("", "", "")).timeout
        
        for attempt in range(max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self._internal_a_stream_run(messages, temperature, max_tokens, **kwargs),
                    timeout=timeout
                )
            except (
                asyncio.TimeoutError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                ConnectionError,
                TimeoutError
            ) as e:
                if attempt < max_retries:
                    print(f"🔄 LLM API调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {type(e).__name__}: {e}")
                    print(f"⏳ 等待 {retry_delay} 秒后重试...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    print(f"❌ LLM API调用最终失败，已重试 {max_retries} 次: {type(e).__name__}: {e}")
                    raise


    async def _internal_a_stream_run(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncResponseStream[str]:
        """
        Internal implementation of async streaming run without retry logic.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            An AsyncResponseStream that yields chunks of the generated content
        """
        processed_messages = self.preprocess_messages(messages)
        
        # Prepare parameters
        params = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # Make API call
        if 'thinking' in params:
            thinking_flag = params.pop('thinking')
            if thinking_flag:
                params['extra_body'] = {"thinking": {"type": "enabled"}}
            else:
                params['extra_body'] = {"thinking": {"type": "disabled"}}

        stream = await self.async_client.chat.completions.create(**params)
        
        # Create async iterator that processes chunks
        async def chunk_iterator() -> AsyncIterator[StreamingChunk[str]]:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                yield self._process_chunk(chunk)
        
        return AsyncResponseStream(
            iterator=chunk_iterator(),
            model_name=self.model_name
        )

GLOBAL_LLM_CONFIG = LLMModelConfig(
    model_name=cfg.llm["model_name"],
    api_key=cfg.llm["api_key"],
    base_url=cfg.llm["base_url"]
)
GLOBAL_LLM = LLMModel(GLOBAL_LLM_CONFIG)

try:
    GLOBAL_THINKING_LLM_CONFIG = LLMModelConfig(
        model_name=cfg.llm_thinking["model_name"],
        api_key=cfg.llm_thinking["api_key"],
        base_url=cfg.llm_thinking["base_url"]
    )
    GLOBAL_THINKING_LLM = LLMModel(GLOBAL_THINKING_LLM_CONFIG)
except Exception as e:
    print(f"加载thinking模型失败，使用llm模型替代: {e}")
    GLOBAL_THINKING_LLM = GLOBAL_LLM

try:
    GLOBAL_VLM_CONFIG = LLMModelConfig(
        model_name=cfg.vlm["model_name"],
        api_key=cfg.vlm["api_key"],
        base_url=cfg.vlm["base_url"]
    )
    GLOBAL_VISION_LLM = LLMModel(GLOBAL_VLM_CONFIG)
except Exception as e:
    print(f"加载vlm模型失败，vision能力不可用: {e}")
    GLOBAL_VISION_LLM = None

if __name__ == "__main__":
    pass
