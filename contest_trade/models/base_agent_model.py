"""
Base Agent Model Module

This module defines the abstract base class for agent models.
All model implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
import asyncio
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Iterator, TypeVar, Generic

T = TypeVar('T')  # Type variable for response content


class ModelResponse(Generic[T]):
    """Base class for model responses."""
    
    def __init__(self, content: T, reasoning_content: T, model_name: str, raw_response: Any = None, proc_response: Any = None):
        """
        Initialize a model response.
        
        Args:
            content: The content of the response
            model_name: The name of the model that generated the response
            raw_response: The raw response from the model provider (optional)
        """
        self.content = content
        self.reasoning_content = reasoning_content
        self.model_name = model_name
        self.raw_response = raw_response
        self.proc_response = proc_response



class StreamingChunk(Generic[T]):
    """Class representing a chunk in a streaming response."""
    
    def __init__(self, content: T, is_finished: bool = False, raw_chunk: Any = None, is_reasoning: bool = False):
        """
        Initialize a streaming chunk.
        
        Args:
            content: The content of the chunk
            is_finished: Whether this is the final chunk
            raw_chunk: The raw chunk from the model provider (optional)
        """
        self.content = content
        self.is_finished = is_finished
        self.raw_chunk = raw_chunk
        self.is_reasoning = is_reasoning


class ResponseStream(Generic[T]):
    """Class for synchronous streaming responses."""
    
    def __init__(self, iterator: Iterator[StreamingChunk[T]], model_name: str):
        """
        Initialize a response stream.
        
        Args:
            iterator: An iterator yielding StreamingChunk objects
            model_name: The name of the model generating the stream
        """
        self._iterator = iterator
        self.model_name = model_name
    
    def __iter__(self) -> Iterator[StreamingChunk[T]]:
        """Return the iterator."""
        return self._iterator


class AsyncResponseStream(Generic[T]):
    """Class for asynchronous streaming responses."""
    
    def __init__(self, iterator: AsyncIterator[StreamingChunk[T]], model_name: str):
        """
        Initialize an async response stream.
        
        Args:
            iterator: An async iterator yielding StreamingChunk objects
            model_name: The name of the model generating the stream
        """
        self._iterator = iterator
        self.model_name = model_name
    
    def __aiter__(self) -> AsyncIterator[StreamingChunk[T]]:
        """Return the async iterator."""
        return self._iterator


class BaseAgentModel(ABC):
    """
    Abstract base class for agent models.
    
    This class defines the interface that all model implementations must follow.
    It follows an async-first design, where the primary implementation is the
    asynchronous streaming method (a_stream_run). All other methods (run, stream_run,
    and a_run) are wrappers around this primary method.
    
    When implementing a subclass, focus on implementing a_stream_run first,
    and the other methods will be handled automatically.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the model.
        
        Args:
            model_name: The name of the model
            **kwargs: Additional model-specific configuration
        """
        self.model_name = model_name
        self.config = kwargs
    
    def run(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse[str]:
        """
        Run the model synchronously and return a complete response.
        
        This is a synchronous wrapper around a_run(). Subclasses should
        implement a_stream_run() as the primary method and all other methods
        will be handled automatically.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            A ModelResponse containing the generated content
        """
        return asyncio.run(self.a_run(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ))

    async def a_run(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        verbose: bool = False,
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
        # Get the stream
        stream = await self.a_stream_run(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
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
        
        # Create a response with the collected content
        return ModelResponse(
            content=self.postprocess_response(full_content),
            reasoning_content=reasoning_content,
            model_name=self.model_name,
            raw_response=raw_chunks if raw_chunks else None
        )
    
    def stream_run(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ResponseStream[str]:
        """
        Run the model synchronously and stream the response.
        
        This is a synchronous wrapper around a_stream_run(). Subclasses should
        implement a_stream_run() as the primary method and this method
        will be handled automatically.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            A ResponseStream that yields chunks of the generated content
        """
        # For synchronous streaming, we need to run the async method in a new event loop
        # and collect all chunks into a list, then yield them one by one
        
        # Run the async method and collect all chunks
        async def collect_chunks():
            chunks = []
            async_stream = await self.a_stream_run(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            async for chunk in async_stream:
                chunks.append(chunk)
            return chunks
            
        # Run the async function and get all chunks
        loop = asyncio.new_event_loop()
        try:
            chunks = loop.run_until_complete(collect_chunks())
        finally:
            loop.close()
            
        # Create a sync iterator that yields the collected chunks
        def sync_iterator():
            for chunk in chunks:
                yield chunk
                
        return ResponseStream(
            iterator=sync_iterator(),
            model_name=self.model_name
        )
    
    @abstractmethod
    async def a_stream_run(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncResponseStream[str]:
        """
        Run the model asynchronously and stream the response.
        
        This is the primary implementation that subclasses must implement.
        All other methods (run, a_run, stream_run) are wrappers around this method.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            An AsyncResponseStream that yields chunks of the generated content
        """
        pass
    
    def preprocess_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Preprocess messages before sending to the model.
        
        This method can be overridden by subclasses to implement custom preprocessing.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Preprocessed list of message dictionaries
        """
        return messages
    
    def postprocess_response(self, response: str) -> str:
        """
        Postprocess the model's response.
        
        This method can be overridden by subclasses to implement custom postprocessing.
        
        Args:
            response: The model's response
            
        Returns:
            Postprocessed response
        """
        return response
