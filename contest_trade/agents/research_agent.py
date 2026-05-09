"""
A general react-agent for analysing.
"""

import os
import re
import sys
import json
import yaml
import textwrap
import asyncio
from loguru import logger
from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from langgraph.graph import StateGraph, END
from utils.llm_utils import count_tokens

from agents.prompts import prompt_for_research_plan, prompt_for_research_choose_tool, prompt_for_research_write_result, prompt_for_research_invest_task, prompt_for_research_invest_output_format
from models.llm_model import GLOBAL_LLM, GLOBAL_THINKING_LLM
from tools.tool_utils import ToolManager, ToolManagerConfig
from config.config import cfg, PROJECT_ROOT
from langchain_core.runnables import RunnableConfig
from utils.market_manager import GLOBAL_MARKET_MANAGER

@dataclass
class ResearchAgentInput:
    """Agent输入"""
    background_information: str
    trigger_time: str


@dataclass
class ResearchAgentOutput:
    """Agent决策结果"""
    task: str  # 任务
    trigger_time: str
    background_information: str
    belief: str
    final_result: str  # 报告
    final_result_thinking: str  # 报告思考

    def to_dict(self):
        return {
            "task": self.task,
            "trigger_time": self.trigger_time,
            "background_information": self.background_information,
            "belief": self.belief,
            "final_result": self.final_result,
            "final_result_thinking": self.final_result_thinking
        }

@dataclass
class ResearchAgentConfig:
    """Agent配置"""
    agent_name: str
    belief: str
    max_react_step: int
    tool_config: ToolManagerConfig
    output_language: str
    plan: bool
    react: bool

    def __init__(self, agent_name: str = "research_agent", belief: str = ""):
        self.agent_name = agent_name
        self.belief = belief
        self.max_react_step = cfg.research_agent_config["max_react_step"]
        self.tool_config = ToolManagerConfig(cfg.research_agent_config["tools"])
        self.output_language = cfg.system_language
        if 'plan' in cfg.research_agent_config:
            self.plan = cfg.research_agent_config["plan"]
        else:
            self.plan = True
        if 'react' in cfg.research_agent_config:
            self.react = cfg.research_agent_config["react"]
        else:
            self.react = True

class ResearchAgentState(TypedDict):
    """LangGraph Agent状态"""
    # 基本信息
    task: str = ""
    trigger_time: str = ""
    belief: str = ""
    background_information: str = ""
    
    # 上下文和预算
    plan_result: str = ""
    tool_call_context: str = ""
    
    # 思考和决策
    selected_tool: dict = {}
    tool_call_count: int = 0
    tool_call_results: list = []
    step_count: int = 0
    
    # 最终结果
    final_result: str = ""
    final_result_thinking: str = ""
    result: ResearchAgentOutput = None


class ResearchAgent:
    """基于LangGraph的投资决策Agent"""
    
    def __init__(self, config: ResearchAgentConfig):
        self.config = config
        self.tool_manager = ToolManager(self.config.tool_config)
        self.app = self._build_graph()
        self.plan = self.config.plan
        self.react = self.config.react


        self.signal_dir = PROJECT_ROOT / "agents_workspace" / "reports" / self.config.agent_name
        if not self.signal_dir.exists():
            self.signal_dir.mkdir(parents=True, exist_ok=True)


    def _build_graph(self) -> StateGraph:
        """构建LangGraph状态图"""
        workflow = StateGraph(ResearchAgentState)
        workflow.add_node("init_signal_dir", self._init_signal_dir)
        workflow.add_node("recompute_signal", self._recompute_signal)
        workflow.add_node("init_data", self._init_data)
        workflow.add_node("plan", self._plan)
        workflow.add_node("tool_selection", self._tool_selection)
        workflow.add_node("call_tool", self._call_tool)
        workflow.add_node("write_result", self._write_result)
        workflow.add_node("submit_result", self._submit_result)
        
        # 定义边
        workflow.set_entry_point("init_signal_dir")
        workflow.add_conditional_edges("init_signal_dir",
            self._recompute_signal,
            {
                "yes": "init_data",
                "no": "submit_result"
            })
        workflow.add_edge("recompute_signal", "init_data")
        workflow.add_conditional_edges("init_data",
            self._need_plan,
            {
                "yes": "plan",
                "no": "tool_selection"
            }
        )
        workflow.add_edge("plan", "tool_selection")
        workflow.add_conditional_edges("tool_selection",
            self._enough_information,
            {
                "enough_information": "write_result",
                "not_enough_information": "call_tool"
            })
        workflow.add_edge("call_tool", "tool_selection")
        workflow.add_edge("write_result", "submit_result")
        workflow.add_edge("submit_result", END)
        return workflow.compile()

    async def _init_signal_dir(self, state: ResearchAgentState) -> ResearchAgentState:
        """try to load signal from file"""
        try:
            signal_file = self.signal_dir / f'{state["trigger_time"].replace(" ", "_").replace(":", "-")}.json'
            if signal_file.exists():
                with open(signal_file, 'r', encoding='utf-8') as f:
                    signal_data = json.load(f)
                state["result"] = ResearchAgentOutput(**signal_data)
        except Exception as e:
            print(f"Error loading signal from file: {e}")
            import traceback
            traceback.print_exc()
        return state
    
    async def _recompute_signal(self, state: ResearchAgentState):
        """recompute signal"""
        if state["result"]:
            print(f"Signal already exists for {state['trigger_time']}, skipping recompute")
            return "no"
        else:
            print(f"Signal does not exist for {state['trigger_time']}, recomputing signal")
            return "yes"

    async def _init_data(self, state: ResearchAgentState) -> ResearchAgentState:
        """初始化数据"""
        state["tool_call_count"] = 0
        return state

    async def _need_plan(self, state: ResearchAgentState) -> str:
        """判断是否需要规划"""
        if self.plan:
            return "yes"
        else:
            return "no"

    async def _plan(self, state: ResearchAgentState) -> ResearchAgentState:
        """规划任务"""
        try:
            if not self.plan:
                state["plan_result"] = ""
                return state
            prompt = prompt_for_research_plan.format(
                current_time=state["trigger_time"],
                task=state["task"],
                background_information=state["background_information"],
                tools_info=self.tool_manager.build_toolcall_context(),
                output_language=self.config.output_language,
            )
            messages = [{"role": "user", "content": prompt}]
            plan_result = await GLOBAL_LLM.a_run(messages, verbose=True, thinking=False, max_retries=10)
            plan_result = plan_result.content
            state["plan_result"] = plan_result.strip()
        except Exception as e:
            logger.error(f"Error in plan: {e}")
            state["plan_result"] = ""
        return state

    async def _tool_selection(self, state: ResearchAgentState) -> ResearchAgentState:
        """选择工具"""
        if not self.react or state["tool_call_count"] >= self.config.max_react_step:
            state["selected_tool"] = {"tool_name": "final_report"}
            return state

        prompt = prompt_for_research_choose_tool.format(
            current_time=state["trigger_time"],
            task=state["task"],
            plan=state["plan_result"],
            background_information=state["background_information"],
            tool_call_context=state["tool_call_context"],
            tools_info=self.tool_manager.build_toolcall_context(),
            output_language=self.config.output_language,
        )
        try:
            next_tool = await self.tool_manager.select_tool_by_llm(
                prompt=prompt,
            )
        except Exception as e:
            logger.error(f"Error in tool_selection: {e}")
            next_tool = {"error": str(e)}
        state["selected_tool"] = next_tool
        return state


    async def _enough_information(self, state: ResearchAgentState) -> str:
        """判断是否足够信息"""
        try:
            estimated_context = prompt_for_research_write_result.format(
                current_time=state["trigger_time"],
                task=state["task"],
                background_information=state["background_information"],
                plan=state["plan_result"],
                tool_call_context=state["tool_call_context"],
                tools_info=self.tool_manager.build_toolcall_context(),
                output_format=self.get_output_format(),
                output_language=self.config.output_language,
            )

            if count_tokens(estimated_context) > 128000:
                return "enough_information"

            selected_tool = state["selected_tool"]
            if state["tool_call_count"] >= self.config.max_react_step:
                return "enough_information"
            if "error" in selected_tool:
                return "not_enough_information"
            if selected_tool["tool_name"] == "final_report":
                return "enough_information"
        except Exception as e:
            logger.error(f"Error in enough_information: {e}")
        return "not_enough_information"


    async def _call_tool(self, state: ResearchAgentState) -> ResearchAgentState:
        """调用工具"""
        selected_tool = state["selected_tool"]
        try:
            if "error" in selected_tool:
                state["tool_call_count"] += 1
                state["tool_call_context"] += json.dumps({
                    "tool_called": selected_tool,
                    "tool_result": selected_tool,
                }, ensure_ascii=False) + "\n"
                return state

            print('Begin to call tool: ', selected_tool)
            tool_name = selected_tool["tool_name"]
            tool_args = selected_tool["properties"]
            tool_result = await self.tool_manager.call_tool(tool_name, tool_args, state["trigger_time"])
            print("tool_result: ", tool_result)
        except Exception as e:
            logger.error(f"Error in call_tool: {e}")
            tool_result = {"error": str(e)}
        
        state["tool_call_count"] += 1
        state["tool_call_context"] += json.dumps({"tool_called":selected_tool,\
                                            "tool_result":tool_result}, ensure_ascii=False) + "\n"
        return state


    async def _write_result(self, state: ResearchAgentState) -> ResearchAgentState:
        """写结果"""
        try:
            if self.get_output_format() is None:
                state["output_format"] = "xxxx"
            prompt = prompt_for_research_write_result.format(
                current_time=state["trigger_time"],
                task=state["task"],
                background_information=state["background_information"],
                plan=state["plan_result"],
                tool_call_context=state["tool_call_context"],
                tools_info=self.tool_manager.build_toolcall_context(),
                output_format=self.get_output_format(),
                output_language=self.config.output_language,
            )
            messages = [{"role": "user", "content": prompt}]
            if cfg.llm_thinking.get("api_key", None):
                result_result = await GLOBAL_THINKING_LLM.a_run(messages, verbose=False, thinking=True, max_retries=5)
            else:
                result_result = await GLOBAL_LLM.a_run(messages, verbose=False, thinking=False, max_retries=5)
            state["final_result"] = result_result.content
            state["final_result_thinking"] = result_result.reasoning_content
            
            # 创建 ResearchAgentOutput 对象
            state["result"] = ResearchAgentOutput(
                task=state["task"],
                trigger_time=state["trigger_time"],
                background_information=state["background_information"],
                belief=state["belief"],
                final_result=state["final_result"],
                final_result_thinking=state["final_result_thinking"]
            )
        except Exception as e:
            logger.error(f"Error in write_report: {e}")
            state["final_result"] = ""
            state["result"] = None
        return state
    
    async def _submit_result(self, state: ResearchAgentState) -> ResearchAgentState:
        """Write the result to a file"""
        try:
            signal_file = self.signal_dir / f'{state["trigger_time"].replace(" ", "_").replace(":", "-")}.json'
            with open(signal_file, 'w', encoding='utf-8') as f:
                json.dump(state["result"].to_dict(), f, ensure_ascii=False, indent=4)
            print(f"Research result saved to {signal_file}")
        except Exception as e:
            print(f"Error writing result: {e}")
            import traceback
            traceback.print_exc()
        return state

    def build_background_information(self, trigger_time: str, belief: str, factors: List):
        """构建背景信息"""
        
        global_market_information = ""
        for factor in factors:
            # 处理不同的factor类型
            if hasattr(factor, 'result') and factor.result:
                factor_output = factor.result
                factor_name = factor_output.agent_name
                factor_update_time = factor_output.trigger_time
                factor_context = factor_output.context_string
            elif hasattr(factor, 'agent_name'):
                factor_name = factor.agent_name
                factor_update_time = factor.trigger_time
                factor_context = factor.context_string
            elif isinstance(factor, dict):
                factor_name = factor.get('agent_name', 'unknown')
                factor_update_time = factor.get('trigger_time', trigger_time)
                factor_context = factor.get('context_string', '')
            else:
                continue
                
            global_market_information += textwrap.dedent(f"""
            <global_summary>
            <source>{factor_name}</source>
            <timestamp>{factor_update_time}</timestamp>
            <content>{factor_context}</content>
            </global_summary>
            """)

        target_market = GLOBAL_MARKET_MANAGER.get_target_symbol_context(trigger_time)
        
        background_information_format = textwrap.dedent("""
        <market_information>
        {global_market_information}
        </market_information>

        <target_market>
        {target_market}
        </target_market>

        <your_belief>
        {belief}
        </your_belief>
        """)
        return background_information_format.format(
            global_market_information=global_market_information,
            target_market=target_market,
            belief=belief
        )

    def get_invest_prompt(self):
        """获取投资提示"""
        return prompt_for_research_invest_task

    def get_output_format(self):
        """获取输出格式"""
        return prompt_for_research_invest_output_format

    async def run_with_monitoring_events(self, input: ResearchAgentInput, config: RunnableConfig = None) -> ResearchAgentOutput:
        """使用事件流监控运行Agent，返回事件流"""
        initial_state = ResearchAgentState(
            trigger_time=input.trigger_time,
            task=self.get_invest_prompt(),
            belief=self.config.belief,
            background_information=input.background_information,
            plan_result="",
            tool_call_context="",
            selected_tool={},
            tool_call_count=0,
            step_count=0,
            final_result="",
            final_result_thinking="",
            result=None
        )
        print(f"🚀 Research Agent Starting - {input.trigger_time}")
        async for event in self.app.astream_events(initial_state, version="v2", config=config or RunnableConfig(recursion_limit=50)):
            yield event

    async def run_with_monitoring(self, input: ResearchAgentInput) -> ResearchAgentOutput:
        """使用事件流监控运行Agent"""
        print(f"🚀 Research Agent Starting - {input.trigger_time}")
        final_result = None
        async for event in self.run_with_monitoring_events(input, RunnableConfig(recursion_limit=50)):
            event_type = event["event"]
            if event_type == "on_chain_start":
                node_name = event["name"]
                if node_name != "__start__":  # 忽略开始事件
                    print(f"🔄 Starting: {node_name}")
                
            elif event_type == "on_chain_end":
                node_name = event["name"]
                if node_name != "__start__":  # 忽略开始事件
                    print(f"✅ Completed: {node_name}")
                    if node_name == "submit_result":
                        final_state = event.get("data", {}).get("output", None)
                        if final_state and "result" in final_state and final_state["result"]:
                            return final_state["result"]
        print(f"✨ Research Agent Completed")
        return final_result
        

if __name__ == "__main__":
    # init instance
    config = ResearchAgentConfig(
        agent_name="research_agent_vtes11",
    )
    agent = ResearchAgent(config)

    #task = input("请输入任务: ")
    task = "新能源龙头股有哪些"
    task = "贵州茅台的最近3天股价"
    agent_input = ResearchAgentInput(
        trigger_time="2025-07-09 09:00:00",
        background_information="123123123"
    )
    agent_output = asyncio.run(agent.run_with_monitoring(agent_input))
    print(agent_output.to_dict())
