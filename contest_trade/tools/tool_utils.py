"""
ToolManager
- Tool Registration
- Tool Calling
- Tool Description
"""
import re
import time
import functools
import asyncio
import importlib
import json
import sys
from pathlib import Path
from loguru import logger
from typing import Dict, List, Any, Callable, Optional, Union, Type
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from models.llm_model import GLOBAL_LLM

class ToolManagerConfig:
    def __init__(self, tool_paths: List[str]):
        self.tool_paths = tool_paths

class ToolManager:
    """Tool registry - instance level, each agent has its own toolset"""
    
    def __init__(self, config: ToolManagerConfig):
        self.tools: Dict[str, Callable] = {}
        
        # ensure relative import support
        PROJECT_ROOT = Path(__file__).parent.parent.resolve()
        sys.path.append(str(PROJECT_ROOT))

        if config.tool_paths:
            self._register_from_configs(config.tool_paths)
        
    def _register_from_configs(self, tool_paths: List):
        """Register tools from config list"""
        for tool_path in tool_paths:
            try:
                self.register_from_module_path(tool_path)
            except Exception as e:
                logger.error(f"Failed to register tool from config {tool_path}: {e}")
    
    def register(self, tool_name: str, tool_function: Callable):
        """Register tool"""
        self.tools[tool_name] = tool_function
    
    def register_function(self, func: Callable, tool_name: str = None):
        """Register function as tool"""
        name = tool_name or getattr(func, 'name', None) or func.__name__
        self.register(name, func)
        return name
    
    def register_from_module_path(self, module_path: str):
        """Register tool function from module path"""
        try:
            parts = module_path.split('.')
            func_name = parts[-1]
            module_name = '.'.join(parts[:-1])
            
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            
            if not callable(func):
                raise ValueError(f"{module_path} is not callable")
            
            return self.register_function(func)
            
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot import {module_path}: {e}")
    
    def register_functions(self, functions: List[Union[Callable, str]]):
        """Batch register tool functions"""
        registered = []
        for func in functions:
            if isinstance(func, str):
                name = self.register_from_module_path(func)
                registered.append(name)
            elif callable(func):
                name = self.register_function(func)
                registered.append(name)
            else:
                logger.warning(f"Skip invalid tool: {func}")
        
        return registered
    
    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """Get tool"""
        return self.tools.get(tool_name)
    
    def get_all_tools(self) -> Dict[str, Callable]:
        """Get all tools"""
        return self.tools.copy()
    
    def build_toolcall_context(self) -> str:
        """Get tools description text"""
        tool_schemas = []
        for tool_name, tool_func in self.tools.items():
            tool_schema = {}
            tool_schema["tool_name"] = tool_name
            tool_schema["description"] = tool_func.description
            for key, value in tool_func.args_schema.model_json_schema().items():
                if key == 'properties':
                    # trigger_time is not needed in tool call
                    if 'trigger_time' in value:
                        value.pop('trigger_time')
                if key == 'required':
                    if 'trigger_time' in value:
                        value.remove('trigger_time')
                if key not in ["title", "type"]:
                    tool_schema[key] = value

            tool_func_dict = tool_func.__dict__
            if 'max_output_len' in tool_func_dict:
                tool_schema["max_output_len"] = tool_func_dict['max_output_len']
            if 'timeout_seconds' in tool_func_dict:
                tool_schema["timeout_seconds"] = tool_func_dict['timeout_seconds']

            tool_schemas.append(tool_schema)
        return json.dumps(tool_schemas, indent=2, ensure_ascii=False)
    
    async def call_tool(self, tool_name: str, kwargs: dict, trigger_time: str=None) -> Any:
        """Call tool"""
        if trigger_time:
            kwargs['trigger_time'] = trigger_time
        tool_func = self.get_tool(tool_name)
        if not tool_func:
            raise ValueError(f"Tool {tool_name} not found")
        
        try:
            print("call tool", tool_name, kwargs)
            if hasattr(tool_func, 'invoke'):
                return await tool_func.ainvoke(kwargs)
            elif asyncio.iscoroutinefunction(tool_func):
                return await tool_func(**kwargs)
            else:
                return tool_func(**kwargs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": "Call tool Failed", "error_msg": str(e)}

    def parse_bounding_json(response: str) -> dict:
        """ parse tool call from llm response """
        bounding_json = re.search(r"<Output>(.*)</Output>", response, flags=re.DOTALL).group(1)
        parsed_output = json.loads(bounding_json)
        assert "tool_name" in parsed_output, "tool_name is required in the output"
        assert "properties" in parsed_output, "properties is required in the output"
        # market in properties
        if "market" in parsed_output["properties"]:
            parsed_output["properties"]["market"] = parsed_output["properties"]["market"].replace(" ", "")
        return parsed_output

    async def select_tool_by_llm(self,
                        prompt: str, 
                        retry_times: int = 3,
                        post_process_func: Callable = parse_bounding_json) -> str:
        """ use llm to select inner tool and return the tool call """
        messages = [{"role": "user", "content": prompt}]
        error_msg = ""
        for i in range(retry_times):
            if error_msg:
                messages.append({"role": "user", "content": error_msg + "\n\n Please try again."})
                error_msg = ""
            try:
                response = await GLOBAL_LLM.a_run(messages, verbose=False, thinking=False, max_retries=5, max_tokens=1000)
                llm_response = response.content
                print("llm_response", llm_response)
                messages.append({"role": "assistant", "content": llm_response})
                parsed_output = post_process_func(llm_response)
                return parsed_output
            except json.JSONDecodeError:
                error_msg += f"Failed to parse tool call {i+1} times"
                print("Failed to parse tool call", llm_response)
            except Exception as e:
                error_msg += f"Failed to call tool {i+1} times with error: {e}"
        return {"error": "Call tool Failed", "error_msg": error_msg}


def _to_str(result: Any) -> str:
    """将任意结果转为字符串"""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception:
        return str(result)


def _truncate(text: str, max_len: int) -> str:
    """截断到最大上下文长度"""
    if text is None:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def smart_tool(
    description: str,
    max_output_len: int = 4000,
    timeout_seconds: float = 60.0,  # 增加默认超时时间
    args_schema: Optional[Type] = None,  # 新增：支持args_schema
):
    """
    智能工具装饰器 - 支持持续状态和参数schema
    
    Args:
        description: tool的描述
        max_output_len: 最大输出上下文长度
        timeout_seconds: 超时时间（秒）
        args_schema: 参数schema类（如PriceInfoInput）
    
    Returns:
        基础格式：
        - 成功：{"success": True, "data": "结果字符串"}
        - 失败：{"success": False, "error_message": "错误信息"}
        
        流式格式（当supports_streaming=True时）：
        - 成功：{"success": True, "data": "结果", "streaming": True, "tools": [...], "observation": "..."}
        - 失败：{"success": False, "error_message": "错误信息"}
    """
    def decorator(func: Callable) -> Callable:
        # 确保是异步函数
        if not asyncio.iscoroutinefunction(func):
            raise ValueError(f"函数 {func.__name__} 必须是异步函数 (async def)")
        
        # 将装饰器参数保存到函数对象上
        func.description = description
        func.max_output_len = max_output_len
        func.timeout_seconds = timeout_seconds
        
        # 根据是否有args_schema选择不同的装饰方式
        if args_schema:
            @tool(description=description, args_schema=args_schema)
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await _process_result(func, args, kwargs, max_output_len, timeout_seconds)
        else:
            @tool(description=description)
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await _process_result(func, args, kwargs, max_output_len, timeout_seconds)
        
        async_wrapper.__dict__.update({
            'max_output_len': max_output_len,
            'timeout_seconds': timeout_seconds
        })
        return async_wrapper
    
    return decorator


async def _process_result(
    func: Callable, 
    args: tuple, 
    kwargs: dict, 
    max_output_len: int, 
    timeout_seconds: float, 
) -> Dict[str, Any]:
    """处理工具执行结果"""
    start_time = time.time()
    
    try:
        # 直接调用异步函数，支持超时
        result = await asyncio.wait_for(
            func(*args, **kwargs), 
            timeout=timeout_seconds
        )
        
        execution_time = time.time() - start_time
        
        if isinstance(result, dict):
            # 流式工具的特殊处理
            if 'tools' in result or 'observation' in result:
                return {
                    "success": True,
                    "data": _to_str(result.get('data', result)),
                    "streaming": True,
                    "tools": result.get('tools', []),
                    "observation": result.get('observation', ""),
                    "metadata": {
                        "execution_time": execution_time,
                        "max_output_len": max_output_len
                    }
                }
        
        # 标准工具处理
        text = _to_str(result)
        text = _truncate(text, max_output_len)
        
        return {
            "success": True,
            "data": text
        }
        
    except asyncio.TimeoutError:
        execution_time = time.time() - start_time
        return {
            "success": False,
            "error_message": f"执行超时（{timeout_seconds}秒）",
        }
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "success": False,
            "error_message": f"执行失败: {str(e)}",
        }

class PrintHelloInput(BaseModel):
    input_string: str = Field(description="string to print.")

@smart_tool(
    description="print a string",
    args_schema=PrintHelloInput,
    max_output_len=1000,
    timeout_seconds=10.0
)
async def print_string(input_string: str):
    return input_string


if __name__ == "__main__":
    """Demo multiple tool registration methods"""

    config = ToolManagerConfig(tool_paths=["tools.tool_utils.print_string"])
    registry = ToolManager(config)

    registry.register_function(print_string)

    registry.register_from_module_path("tools.tool_utils.print_string")

    # get tools description
    print(registry.build_toolcall_context())

    # list all tools
    print(registry.get_all_tools())

    # call tool
    result = asyncio.run(registry.call_tool("print_string", {"input_string": "Hello World!"}))
    print(result)

