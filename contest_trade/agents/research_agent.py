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
        if not self.react:
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
            # 检查是否有成功的工具调用
            tool_call_context = state["tool_call_context"]
            has_successful_calls = False
            if tool_call_context:
                import json
                lines = tool_call_context.strip().split('\n')
                for line in lines:
                    if line.strip():
                        try:
                            call_data = json.loads(line)
                            if call_data.get("tool_result", {}).get("status") == "success":
                                has_successful_calls = True
                                break
                        except:
                            continue
            
            print(f"🔍 [Step {state['tool_call_count']}] Checking if enough information:")
            print(f"   - Tool calls made: {state['tool_call_count']}/{self.config.max_react_step}")
            print(f"   - Has successful calls: {has_successful_calls}")
            
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
                print("   - Stopping: Context too long")
                return "enough_information"

            selected_tool = state["selected_tool"]
            if "error" in selected_tool:
                print("   - Continuing: Tool selection has error")
                return "not_enough_information"
            
            # 如果调用了final_report或达到最大步数，结束
            if selected_tool["tool_name"] == "final_report" or \
                state["tool_call_count"] >= self.config.max_react_step:
                print(f"   - Stopping: final_report called or max steps reached")
                return "enough_information"
                
            # 如果没有成功的工具调用且还有剩余步数，继续尝试
            if not has_successful_calls and state["tool_call_count"] < self.config.max_react_step:
                print("   - Continuing: No successful tool calls yet, need more data")
                return "not_enough_information"
                
            print("   - Continuing: Default behavior")
            return "not_enough_information"
            
        except Exception as e:
            logger.error(f"Error in enough_information: {e}")
            return "not_enough_information"


    async def _call_tool(self, state: ResearchAgentState) -> ResearchAgentState:
        """调用工具"""
        selected_tool = state["selected_tool"]
        try:
            print(f'🔧 [Step {state["tool_call_count"] + 1}] Begin to call tool: {selected_tool}')
            tool_name = selected_tool["tool_name"]
            tool_args = selected_tool["properties"]
            tool_result = await self.tool_manager.call_tool(tool_name, tool_args, state["trigger_time"])
            
            # 详细检查工具调用结果
            if tool_result is None:
                print(f"⚠️  WARNING: Tool '{tool_name}' returned None - possible tool failure")
                tool_result = {"error": f"Tool {tool_name} returned None", "status": "failed", "has_data": False}
            elif isinstance(tool_result, str) and len(tool_result.strip()) == 0:
                print(f"⚠️  WARNING: Tool '{tool_name}' returned empty string")
                tool_result = {"error": f"Tool {tool_name} returned empty result", "status": "failed", "has_data": False}
            elif isinstance(tool_result, dict) and "error" in tool_result:
                print(f"❌ Tool '{tool_name}' returned error: {tool_result.get('error')}")
                tool_result["status"] = "failed"
                tool_result["has_data"] = False
            else:
                print(f"✅ Tool '{tool_name}' executed successfully")
                print(f"📊 Result type: {type(tool_result)}, length: {len(str(tool_result))}")
                # 确保有数据标记
                if isinstance(tool_result, dict):
                    tool_result["status"] = "success"
                    tool_result["has_data"] = True
                else:
                    tool_result = {"data": tool_result, "status": "success", "has_data": True}
                
        except Exception as e:
            print(f"❌ CRITICAL: Tool '{selected_tool.get('tool_name', 'unknown')}' execution failed: {e}")
            logger.error(f"Error in call_tool: {e}")
            tool_result = {"error": str(e), "status": "failed", "has_data": False}
        
        state["tool_call_count"] += 1
        state["tool_call_context"] += json.dumps({"tool_called":selected_tool,\
                                            "tool_result":tool_result}, ensure_ascii=False) + "\n"
        return state


    def _calculate_enhanced_data_quality(self, tool_call_context: str) -> dict:
        """
        计算增强的数据质量指标
        """
        import json
        
        # 基础成功率计算
        successful_calls = 0
        total_calls = 0
        successful_tools = []
        data_sources = set()
        tool_results = []
        content_quality_score = 0.0
        
        if tool_call_context:
            lines = tool_call_context.strip().split('\n')
            for line in lines:
                if line.strip():
                    try:
                        call_data = json.loads(line)
                        total_calls += 1
                        
                        if call_data.get("tool_result", {}).get("status") == "success":
                            successful_calls += 1
                            tool_name = call_data.get("tool_called", {}).get("tool_name", "unknown")
                            successful_tools.append(tool_name)
                            data_sources.add(tool_name)
                            
                            # 评估内容质量
                            result = call_data.get("tool_result", {})
                            result_length = len(str(result))
                            
                            if result_length > 100:  # 有实质性内容
                                content_quality_score += 1
                            elif result_length > 50:  # 有基本内容
                                content_quality_score += 0.5
                                
                            tool_results.append({
                                'tool': tool_name,
                                'result_length': result_length,
                                'has_structured_data': isinstance(result.get('data'), (dict, list))
                            })
                    except:
                        continue
        
        # 计算各项指标
        success_rate = successful_calls / total_calls if total_calls > 0 else 0
        content_quality = content_quality_score / total_calls if total_calls > 0 else 0
        source_diversity = len(data_sources) / max(total_calls, 1)
        data_freshness = min(successful_calls / max(total_calls, 1), 1.0)
        
        # 综合质量分数
        overall_score = (
            success_rate * 0.4 +           # 成功率权重40%
            content_quality * 0.3 +        # 内容质量权重30%
            source_diversity * 0.2 +       # 数据源多样性权重20%
            data_freshness * 0.1           # 数据时效性权重10%
        )
        
        return {
            'success_rate': success_rate,
            'content_quality': content_quality,
            'source_diversity': source_diversity,
            'data_freshness': data_freshness,
            'overall_score': overall_score,
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'unique_sources': len(data_sources),
            'successful_tools': successful_tools,
            'tool_results': tool_results
        }
    
    def _generate_enhanced_hallucination_warning(self, quality_metrics: dict) -> str:
        """
        基于增强的数据质量指标生成精确的幻觉防护警告
        """
        overall_score = quality_metrics['overall_score']
        success_rate = quality_metrics['success_rate']
        content_quality = quality_metrics['content_quality']
        source_diversity = quality_metrics['source_diversity']
        successful_calls = quality_metrics['successful_calls']
        
        if overall_score < 0.3:
            return f"""

🚨 严重警告：数据质量极低 (总分: {overall_score:.2f})
**严格限制要求**：
- 成功率: {success_rate:.1%} | 内容质量: {content_quality:.2f} | 数据源多样性: {source_diversity:.2f}
- **绝对禁止**：编造任何数据、推测未验证信息、使用模糊表述
- **必须做到**：每个结论都标注"[数据不足]"，概率评估不得超过30%
- **强制要求**：在limitations中详细说明数据缺失情况
- **输出限制**：只能基于{successful_calls}个成功工具调用的确切结果
"""
        elif overall_score < 0.5:
            return f"""

⚠️ 重要警告：数据质量较低 (总分: {overall_score:.2f})
**限制要求**：
- 成功率: {success_rate:.1%} | 内容质量: {content_quality:.2f} | 数据源多样性: {source_diversity:.2f}
- **严格禁止**：编造数据、过度推测、使用不确定的表述如"可能"、"据说"
- **必须标注**：每个证据的确定性级别不得超过60%
- **强制引用**：使用标准格式 [工具名|时间|具体数值] 引用所有数据
- **概率限制**：最终概率评估不得超过50%
"""
        elif overall_score < 0.7:
            return f"""

⚠️ 注意：数据完整性中等 (总分: {overall_score:.2f})
**谨慎要求**：
- 成功率: {success_rate:.1%} | 内容质量: {content_quality:.2f} | 数据源多样性: {source_diversity:.2f}
- **禁止行为**：编造具体数值、混淆事实与推论
- **必须区分**：明确标注哪些是直接数据，哪些是基于数据的推论
- **引用要求**：所有关键数据必须使用标准引用格式
- **不确定性**：适当标注不确定性级别，保持谨慎态度
"""
        else:
            return f"""

✅ 数据质量良好 (总分: {overall_score:.2f})
**标准要求**：
- 成功率: {success_rate:.1%} | 内容质量: {content_quality:.2f} | 数据源多样性: {source_diversity:.2f}
- **基本原则**：确保所有结论都有明确的数据支撑
- **引用标准**：使用规范的数据引用格式
- **客观性**：区分客观事实和主观分析
- **完整性**：在data_quality_assessment中提供详细的质量评估
"""

    async def _write_result(self, state: ResearchAgentState) -> ResearchAgentState:
        """写结果 - 增强版幻觉检测"""
        try:
            # 增强版幻觉检测：多层数据验证
            tool_call_context = state["tool_call_context"]
            print(f"📊 Analyzing tool call context length: {len(tool_call_context)}")
            
            # 计算增强的数据质量指标
            quality_metrics = self._calculate_enhanced_data_quality(tool_call_context)
            
            print(f"📈 Enhanced Data Quality Metrics:")
            print(f"   - Overall Score: {quality_metrics['overall_score']:.2f}")
            print(f"   - Success Rate: {quality_metrics['success_rate']:.1%}")
            print(f"   - Content Quality: {quality_metrics['content_quality']:.2f}")
            print(f"   - Source Diversity: {quality_metrics['source_diversity']:.2f}")
            print(f"   - Successful Tools: {quality_metrics['successful_tools']}")
            
            # 生成精确的幻觉防护警告
            hallucination_warning = self._generate_enhanced_hallucination_warning(quality_metrics)
            
            # 将质量指标添加到状态中，供后续使用
            state["data_quality_metrics"] = quality_metrics
            
            if self.get_output_format() is None:
                state["output_format"] = "xxxx"
            prompt = prompt_for_research_write_result.format(
                current_time=state["trigger_time"],
                task=state["task"],
                background_information=state["background_information"],
                plan=state["plan_result"],
                tool_call_context=state["tool_call_context"] + hallucination_warning,
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
    task = "美股科技龙头股有哪些"
    task = "苹果公司的最近3天股价"
    agent_input = ResearchAgentInput(
        trigger_time="2025-07-09 09:00:00",
        background_information="123123123"
    )
    agent_output = asyncio.run(agent.run_with_monitoring(agent_input))
    print(agent_output.to_dict())

