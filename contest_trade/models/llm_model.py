"""
Multi-Provider LLM Model Implementation

This module provides an implementation of the BaseAgentModel that supports
multiple LLM providers: OpenAI, Gemini, and Ollama.
It follows an async-first approach, where the primary implementation is the
asynchronous streaming method (a_stream_run).
"""
import os
import sys
import httpx
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, AsyncIterator, Callable, Any
from enum import Enum
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from config.config import cfg

from models.provider_auth import (
    format_missing_key_error,
    is_openai_compatible,
    normalize_secret,
    normalize_temperature,
    resolve_api_key,
    resolve_base_url,
)
from models.base_agent_model import (
    BaseAgentModel,
    AsyncResponseStream,
    StreamingChunk,
    ModelResponse
)


class ProviderType(Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"


def _normalize_provider(provider: str) -> str:
    """Map provider aliases to transport provider."""
    normalized = (provider or "openai").lower().strip()
    if is_openai_compatible(normalized):
        return ProviderType.OPENAI.value
    if normalized == ProviderType.GEMINI.value:
        return ProviderType.GEMINI.value
    if normalized == ProviderType.OLLAMA.value:
        return ProviderType.OLLAMA.value
    return normalized

class LLMModelConfig:
    def __init__(self, provider: str, model_name: str, api_key: str = None, base_url: str = None,
                 max_retries: int = 3, retry_delay: float = 3.0, timeout: float = 60.0,
                 extra_headers: dict = None, proxys: dict = None, **kwargs):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.extra_headers = extra_headers
        self.proxys = proxys
        # Store additional provider-specific configuration
        self.extra_config = kwargs


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMModelConfig):
        self.config = config
        self.model_name = config.model_name
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.extra_headers = config.extra_headers
        self.proxys = config.proxys
    
    @abstractmethod
    async def create_stream(self, messages: List[Dict[str, str]], temperature: float, 
                           max_tokens: Optional[int], **kwargs) -> Any:
        """Create a streaming response from the provider."""
        pass
    
    @abstractmethod
    def process_chunk(self, chunk: Any) -> StreamingChunk[str]:
        """Process a streaming chunk from the provider."""
        pass
    
    def preprocess_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Preprocess messages for the provider. Can be overridden by subclasses."""
        return messages


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, config: LLMModelConfig):
        super().__init__(config)
        
        # Import OpenAI modules
        try:
            import openai
            from openai import OpenAI, AsyncOpenAI
            from openai.types.chat import ChatCompletionChunk
            self._openai = openai
            self._ChatCompletionChunk = ChatCompletionChunk
        except ImportError:
            raise ImportError("openai package is required for OpenAI provider. Install with: pip install openai")
        
        # Resolve credentials: config -> provider env vars -> generic fallback
        self.api_key = resolve_api_key(
            config.provider,
            config_key=self.api_key,
            base_url=self.base_url,
        )
        self.base_url = resolve_base_url(config.provider, config_base_url=self.base_url)
        
        # Initialize clients (lazy initialization if no API key)
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=httpx.Client(proxy=self.proxys) if self.proxys else None
            )
            
            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=httpx.AsyncClient(proxy=self.proxys) if self.proxys else None
            )
            
            if self.extra_headers is not None:
                self.client = self.client.with_options(default_headers=self.extra_headers)
                self.async_client = self.async_client.with_options(default_headers=self.extra_headers)
        else:
            # Defer client initialization until first use
            self.client = None
            self.async_client = None
    
    def _ensure_clients(self):
        """Ensure OpenAI clients are initialized."""
        if self.async_client is None:
            if not self.api_key:
                raise ValueError(format_missing_key_error(
                    self.config.provider,
                    base_url=self.base_url,
                ))
            
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=httpx.Client(proxy=self.proxys) if self.proxys else None
            )
            
            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=httpx.AsyncClient(proxy=self.proxys) if self.proxys else None
            )
            
            if self.extra_headers is not None:
                self.client = self.client.with_options(default_headers=self.extra_headers)
                self.async_client = self.async_client.with_options(default_headers=self.extra_headers)
    
    async def create_stream(self, messages: List[Dict[str, str]], temperature: float, 
                           max_tokens: Optional[int], **kwargs) -> Any:
        self._ensure_clients()  # Ensure clients are initialized
        processed_messages = self.preprocess_messages(messages)
        temperature = normalize_temperature(self.base_url, temperature)
        
        params = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # Handle thinking mode for compatible models
        if 'thinking' in params:
            thinking_flag = params.pop('thinking')
            if thinking_flag:
                params['extra_body'] = {"thinking": {"type": "enabled"}}
            else:
                params['extra_body'] = {"thinking": {"type": "disabled"}}
        
        return await self.async_client.chat.completions.create(**params)
    
    def process_chunk(self, chunk) -> StreamingChunk[str]:
        """Process a streaming chunk from OpenAI."""
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


class GeminiProvider(BaseProvider):
    """Google Gemini provider implementation."""
    
    def __init__(self, config: LLMModelConfig):
        super().__init__(config)
        
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError("google-generativeai package is required for Gemini provider. Install with: pip install google-generativeai")
        
        self.api_key = resolve_api_key(
            ProviderType.GEMINI.value,
            config_key=self.api_key,
            base_url=self.base_url,
        )
        if not self.api_key:
            raise ValueError(format_missing_key_error(
                ProviderType.GEMINI.value,
                base_url=self.base_url,
                config_path="llm.api_key",
            ))

        genai.configure(api_key=self.api_key)
        
        # Configure model
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )
    
    def preprocess_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert OpenAI format messages to Gemini format."""
        gemini_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Map OpenAI roles to Gemini roles
            if role == "system":
                # Gemini doesn't have system role, prepend to first user message
                if gemini_messages and gemini_messages[-1].get("role") == "user":
                    gemini_messages[-1]["parts"] = [f"{content}\n\n{gemini_messages[-1]['parts'][0]}"]
                else:
                    gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})
        
        return gemini_messages
    
    async def create_stream(self, messages: List[Dict[str, str]], temperature: float, 
                           max_tokens: Optional[int], **kwargs) -> Any:
        processed_messages = self.preprocess_messages(messages)
        
        # Update generation config
        generation_config = self.model._generation_config.copy()
        generation_config["temperature"] = temperature
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        
        # Create chat session
        chat = self.model.start_chat(history=processed_messages[:-1] if len(processed_messages) > 1 else [])
        
        # Get the last message content
        last_message = processed_messages[-1]["parts"][0] if processed_messages else ""
        
        # Generate streaming response
        response = await chat.send_message_async(last_message, stream=True)
        return response
    
    def process_chunk(self, chunk) -> StreamingChunk[str]:
        """Process a streaming chunk from Gemini."""
        content = ""
        is_finished = False
        
        if hasattr(chunk, 'text'):
            content = chunk.text
        
        # Gemini doesn't provide explicit finish signals in chunks
        # We'll rely on the async iterator to determine when finished
        
        return StreamingChunk(
            content=content,
            is_finished=is_finished,
            raw_chunk=chunk,
            is_reasoning=False
        )


class OllamaProvider(BaseProvider):
    """Ollama provider implementation."""
    
    def __init__(self, config: LLMModelConfig):
        super().__init__(config)
        
        self.base_url = resolve_base_url(
            ProviderType.OLLAMA.value,
            config_base_url=self.base_url,
        ) or "http://localhost:11434"
        self.api_key = resolve_api_key(
            ProviderType.OLLAMA.value,
            config_key=self.api_key,
            base_url=self.base_url,
        )
    
    async def create_stream(self, messages: List[Dict[str, str]], temperature: float, 
                           max_tokens: Optional[int], **kwargs) -> Any:
        processed_messages = self.preprocess_messages(messages)
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.extra_headers:
            headers.update(self.extra_headers)
        
        payload = {
            "model": self.model_name,
            "messages": processed_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        # Add any extra options from config
        if self.config.extra_config:
            payload["options"].update(self.config.extra_config)
        
        # Make HTTP request to Ollama API
        async with httpx.AsyncClient(proxy=self.proxys) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers=headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.aiter_lines()
    
    def process_chunk(self, chunk_line: str) -> StreamingChunk[str]:
        """Process a streaming chunk from Ollama."""
        import json
        
        try:
            chunk_data = json.loads(chunk_line)
            content = chunk_data.get("message", {}).get("content", "")
            is_finished = chunk_data.get("done", False)
            
            return StreamingChunk(
                content=content,
                is_finished=is_finished,
                raw_chunk=chunk_data,
                is_reasoning=False
            )
        except json.JSONDecodeError:
            # Skip malformed JSON lines
            return StreamingChunk(
                content="",
                is_finished=False,
                raw_chunk=chunk_line,
                is_reasoning=False
            )


def create_provider(config: LLMModelConfig) -> BaseProvider:
    """Factory function to create the appropriate provider based on config."""
    provider_type = _normalize_provider(config.provider)

    if provider_type == ProviderType.OPENAI.value:
        return OpenAIProvider(config)
    elif provider_type == ProviderType.GEMINI.value:
        return GeminiProvider(config)
    elif provider_type == ProviderType.OLLAMA.value:
        return OllamaProvider(config)
    else:
        supported = sorted({p.value for p in ProviderType} | {"deepseek", "stepfun", "moonshot", "qwen", "zhipu"})
        raise ValueError(f"Unsupported provider: {config.provider}. Supported providers: {supported}")


class LLMModel(BaseAgentModel):
    """
    Multi-provider LLM model implementation.
    
    This class provides a concrete implementation of the BaseAgentModel that supports
    multiple LLM providers (OpenAI, Gemini, Ollama).
    Following the async-first approach, it delegates to the appropriate provider
    for the actual implementation.
    """
    
    def __init__(
        self,
        config: LLMModelConfig,
        **kwargs
    ):
        """
        Initialize the LLM model with the specified provider.
        
        Args:
            config: LLMModelConfig instance specifying provider and configuration
            **kwargs: Additional configuration parameters
        """
        super().__init__(config.model_name, **kwargs)
        
        self.config = config
        self.model_name = config.model_name
        
        # Create the appropriate provider
        self.provider = create_provider(config)
    
    def preprocess_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Preprocess messages using the provider's preprocessing."""
        return self.provider.preprocess_messages(messages)

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
                ConnectionError,
                TimeoutError
            ) as e:
                # Handle provider-specific exceptions
                should_retry = False
                if self.config.provider == ProviderType.OPENAI.value:
                    try:
                        import openai
                        if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError)):
                            should_retry = True
                    except ImportError:
                        pass
                
                # Always retry on these common exceptions
                if isinstance(e, (asyncio.TimeoutError, ConnectionError, TimeoutError)):
                    should_retry = True
                
                if should_retry and attempt < max_retries:
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
        # Get the stream from the provider
        stream = await self.provider.create_stream(messages, temperature, max_tokens, **kwargs)
        
        # Create async iterator that processes chunks using provider's chunk processor
        async def chunk_iterator() -> AsyncIterator[StreamingChunk[str]]:
            if self.config.provider == ProviderType.OLLAMA.value:
                # Handle Ollama's line-by-line streaming
                async for line in stream:
                    if line.strip():
                        chunk = self.provider.process_chunk(line)
                        if chunk.content or chunk.is_finished:
                            yield chunk
            elif self.config.provider == ProviderType.GEMINI.value:
                # Handle Gemini streaming
                async for chunk in stream:
                    processed_chunk = self.provider.process_chunk(chunk)
                    if processed_chunk.content:
                        yield processed_chunk
            else:
                # Handle OpenAI and OpenAI-compatible streaming
                async for chunk in stream:
                    if hasattr(chunk, 'choices') and not chunk.choices:
                        continue
                    yield self.provider.process_chunk(chunk)
        
        return AsyncResponseStream(
            iterator=chunk_iterator(),
            model_name=self.model_name
        )

# Helper function to determine provider from model name or base URL
def detect_provider(model_name: str, base_url: str = None) -> str:
    """Auto-detect provider based on model name or base URL."""
    model_name = model_name.lower()
    
    # Check for Gemini models
    if "gemini" in model_name or "google" in model_name:
        return ProviderType.GEMINI.value
    
    # Check for Ollama (usually local models or specific patterns)
    if base_url and ("localhost" in base_url or "127.0.0.1" in base_url or ":11434" in base_url):
        return ProviderType.OLLAMA.value
    
    # Check for common Ollama model names
    ollama_models = ["llama", "mistral", "codellama", "vicuna", "alpaca", "orca", "phi", "neural-chat"]
    if any(ollama_model in model_name for ollama_model in ollama_models):
        return ProviderType.OLLAMA.value
    
    # Default to OpenAI (includes OpenAI-compatible APIs)
    return ProviderType.OPENAI.value


def _build_llm_config(section: dict, fallback_section: dict = None) -> LLMModelConfig:
    """Build LLMModelConfig with provider-aware credential resolution."""
    provider = section.get("provider", detect_provider(section["model_name"], section.get("base_url")))
    base_url = section.get("base_url")
    api_key = resolve_api_key(
        provider,
        config_key=normalize_secret(section.get("api_key")),
        base_url=base_url,
    )
    if api_key is None and fallback_section:
        api_key = resolve_api_key(
            fallback_section.get("provider", provider),
            config_key=normalize_secret(fallback_section.get("api_key")),
            base_url=fallback_section.get("base_url"),
        )
    return LLMModelConfig(
        provider=provider,
        model_name=section["model_name"],
        api_key=api_key,
        base_url=resolve_base_url(provider, base_url) or base_url,
    )


# Create global configurations with auto-detected providers
GLOBAL_LLM_CONFIG = _build_llm_config(cfg.llm)
GLOBAL_LLM = LLMModel(GLOBAL_LLM_CONFIG)

try:
    GLOBAL_THINKING_LLM_CONFIG = _build_llm_config(cfg.llm_thinking, fallback_section=cfg.llm)
    if GLOBAL_THINKING_LLM_CONFIG.api_key is None:
        raise ValueError(format_missing_key_error(
            GLOBAL_THINKING_LLM_CONFIG.provider,
            base_url=GLOBAL_THINKING_LLM_CONFIG.base_url,
            config_path="llm_thinking.api_key",
        ))
    GLOBAL_THINKING_LLM = LLMModel(GLOBAL_THINKING_LLM_CONFIG)
except Exception as e:
    print(f"加载thinking模型失败，使用llm模型替代: {e}")
    GLOBAL_THINKING_LLM = GLOBAL_LLM

try:
    GLOBAL_VLM_CONFIG = _build_llm_config(cfg.vlm, fallback_section=cfg.llm)
    if GLOBAL_VLM_CONFIG.api_key is None:
        raise ValueError(format_missing_key_error(
            GLOBAL_VLM_CONFIG.provider,
            base_url=GLOBAL_VLM_CONFIG.base_url,
            config_path="vlm.api_key",
        ))
    GLOBAL_VISION_LLM = LLMModel(GLOBAL_VLM_CONFIG)
except Exception as e:
    print(f"加载vlm模型失败，vision能力不可用: {e}")
    GLOBAL_VISION_LLM = None

if __name__ == "__main__":
    pass
