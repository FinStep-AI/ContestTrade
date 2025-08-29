"""
Simplified Trade Company - 合并所有冗余代码，包装成LangGraph工作流
"""
import re
import json
import asyncio
from typing import List, Dict, TypedDict
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import dispatch_custom_event

from config.config import cfg, PROJECT_ROOT
from agents.data_analysis_agent import DataAnalysisAgent, DataAnalysisAgentConfig, DataAnalysisAgentInput
from agents.research_agent import ResearchAgent, ResearchAgentConfig, ResearchAgentInput
from contest.judger_data_converter import DataFormatConverter
from contest.judger_executor import run_judger_critic_pipeline
from contest.judger_executor import get_signal_details, format_signal_output
from utils.market_manager import GLOBAL_MARKET_MANAGER

# 统一的状态定义
class CompanyState(TypedDict):
    trigger_time: str
    data_factors: List[Dict]
    research_signals: List[Dict]
    judger_scores: Dict
    optimized_weights: Dict
    all_events: List[Dict]
    step_results: Dict

class SimpleTradeCompany:
    def __init__(self):
        # 设置工作目录
        self.workspace_dir = str(PROJECT_ROOT / "agents_workspace")
        
        # 初始化Data Agents
        self.data_agents = {}
        for agent_config_idx, agent_config in enumerate(cfg.data_agents_config):
            custom_config = DataAnalysisAgentConfig(
                source_list=agent_config["data_source_list"],
                agent_name=agent_config["agent_name"],
                final_target_tokens=agent_config.get("final_target_tokens", 4000),
                bias_goal=agent_config.get("bias_goal", ""),
            )
            self.data_agents[agent_config_idx] = DataAnalysisAgent(custom_config)
        
        # 初始化Research Agents
        self.research_agents = {}
        
        # 从belief_list.json读取belief配置
        belief_list_path = PROJECT_ROOT / cfg.research_agent_config["belief_list_path"]
        with open(belief_list_path, 'r', encoding='utf-8') as f:
            belief_list = json.load(f)
        
        for agent_config_idx, belief in enumerate(belief_list):
            custom_config = ResearchAgentConfig(
                agent_name=f"agent_{agent_config_idx}",
                belief=belief,
            )
            self.research_agents[agent_config_idx] = ResearchAgent(custom_config)
        
        # 初始化数据转换器
        self.data_converter = DataFormatConverter(self.workspace_dir)

    # LangGraph节点函数
    async def run_data_agents_step(self, state: CompanyState, config: RunnableConfig) -> CompanyState:
        """运行Data Agents步骤"""
        trigger_time = state["trigger_time"]
        
        print("🚀 开始并发运行Data Agents...")
        
        # 创建并发任务
        agent_tasks = []
        for agent_id, agent in self.data_agents.items():
            task = self._run_single_data_agent(agent_id, agent, trigger_time, config)
            agent_tasks.append(task)
        
        # 并发执行
        results = await asyncio.gather(*agent_tasks)
        
        # 收集结果
        all_factors = []
        all_events = []
        for result in results:
            if result:
                all_factors.append(result["factor"])
                all_events.extend(result["events"])
        
        print(f"✅ Data Agents完成，有效结果: {len(all_factors)}")
        
        # 更新状态
        all_events_state = state["all_events"].copy()
        all_events_state.extend(all_events)
        
        step_results = state["step_results"].copy()
        step_results["data_team"] = {"factors_count": len(all_factors), "events_count": len(all_events)}
        
        return {
            "data_factors": all_factors,
            "all_events": all_events_state,
            "step_results": step_results
        }

    async def run_research_agents_step(self, state: CompanyState, config: RunnableConfig) -> CompanyState:
        """运行Research Agents步骤"""
        trigger_time = state["trigger_time"]
        data_factors = state["data_factors"]
        
        print("🚀 开始并发运行Research Agents...")
        
        # 创建并发任务
        agent_tasks = []
        for agent_id, agent in self.research_agents.items():
            task = self._run_single_research_agent(agent_id, agent, trigger_time, data_factors, config)
            agent_tasks.append(task)
        
        # 并发执行 - 使用return_exceptions=True确保一个agent失败不会影响其他agent
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # 收集结果
        all_signals = []
        all_events = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Research Agent {i} 执行异常: {str(result)}")
                continue
            if result and result["signals"]:
                all_signals.extend(result["signals"])
                all_events.extend(result["events"])
        
        print(f"✅ Research Agents完成，有效信号总数: {len(all_signals)}")
        
        # 更新状态
        all_events_state = state["all_events"].copy()
        all_events_state.extend(all_events)
        
        step_results = state["step_results"].copy()
        step_results["research_team"] = {"signals_count": len(all_signals), "events_count": len(all_events)}
        
        return {
            "research_signals": all_signals,
            "all_events": all_events_state,
            "step_results": step_results
        }

    async def run_judger_critic_step(self, state: CompanyState, config: RunnableConfig) -> CompanyState:
        """运行JudgerCritic步骤 - 调用子脚本函数"""
        trigger_time = state["trigger_time"]
        if not cfg.researcher_contest_config["contest_mode"]:
            return {
                "judger_scores": {},
                "optimized_weights": {},
                "step_results": state["step_results"]
            }
        
        # 调用子脚本中的JudgerCritic执行器
        judger_result = await run_judger_critic_pipeline(
            trigger_time=trigger_time,
            workspace_dir=self.workspace_dir,
            research_agents=self.research_agents
        )
        
        # 更新状态
        step_results = state["step_results"].copy()
        
        if judger_result['status'] == 'success':
            judger_scores = judger_result.get('consensus_scores', {})
            optimized_weights = judger_result.get('optimized_weights', {})
            
            step_results["judger_critic"] = {
                "status": "success",
                "scores_count": len(judger_scores),
                "weights_count": len(optimized_weights),
                "avg_score": sum(judger_scores.values()) / len(judger_scores) if judger_scores else 0,
                "consensus_scores": judger_scores,
                "optimized_weights": optimized_weights
            }
            
            print(f"✅ JudgerCritic步骤完成，评分信号数量: {len(judger_scores)}")
            
            return {
                "judger_scores": judger_scores,
                "optimized_weights": optimized_weights,
                "step_results": step_results
            }
        else:
            step_results["judger_critic"] = {
                "status": "failed",
                "reason": judger_result.get('reason', 'unknown')
            }
            
            print(f"❌ JudgerCritic步骤失败: {judger_result.get('reason', 'unknown')}")
            
            return {
                "judger_scores": {},
                "optimized_weights": {},
                "step_results": step_results
            }

    async def run_contest_step(self, state: CompanyState, config: RunnableConfig) -> CompanyState:
        """运行竞赛步骤"""
        research_signals = state["research_signals"]
        judger_scores = state["judger_scores"]
        optimized_weights = state["optimized_weights"]
        
        print("🚀 开始运行竞赛步骤...")
        
        # 基于权重选择最佳信号
        if optimized_weights:
            # 根据权重排序选择前3个信号
            sorted_weights = sorted(optimized_weights.items(), key=lambda x: x[1], reverse=True)
            best_signal_names = [name for name, weight in sorted_weights[:3]]
            
            # 从research_signals中找对应的信号
            best_signals = []
            for signal in research_signals:
                if signal.get('agent_name') in best_signal_names or signal.get('agent_id') in [int(name.split('_')[-1]) if '_' in name else -1 for name in best_signal_names]:
                    best_signals.append(signal)
        else:
            # 简单的竞赛逻辑：取前所有作为最佳信号
            best_signals = research_signals if research_signals else []
        
        print(f"✅ 竞赛步骤完成，最佳信号数量: {len(best_signals)}")
        
        # 更新状态
        step_results = state["step_results"].copy()
        step_results["contest"] = {
            "best_signals_count": len(best_signals), 
            "best_signals": best_signals,
            "used_weights": bool(optimized_weights)
        }
        
        return {
            "step_results": step_results
        }

    async def finalize_step(self, state: CompanyState, config: RunnableConfig) -> CompanyState:
        """最终结果步骤"""
        trigger_time = state["trigger_time"]
        data_factors = state["data_factors"]
        research_signals = state["research_signals"]
        all_events = state["all_events"]
        step_results = state["step_results"]
        
        print("🚀 开始最终结果步骤...")
        
        # 获取最佳信号
        best_signals = step_results.get("contest", {}).get("best_signals", [])
        
        # 生成最终结果
        final_result = {
            "trigger_time": trigger_time,
            "data_factors_count": len(data_factors),
            "research_signals_count": len(research_signals),
            "total_events_count": len(all_events),
            "best_signals": best_signals,
            "step_results": step_results
        }

        print("✅ 最终结果步骤完成")
        
        return {
            "step_results": step_results
        }

    # 辅助函数
    async def _run_single_data_agent(self, agent_id: int, agent, trigger_time: str, config: RunnableConfig):
        """运行单个data agent"""
        print(f"🔍 开始运行Data Agent {agent_id} ({agent.config.agent_name})...")
        
        agent_input = DataAnalysisAgentInput(trigger_time=trigger_time)
        agent_events = []
        agent_output = None
        
        # 运行agent并收集事件
        async for event in agent.run_with_monitoring_events(agent_input, config):
            # 转发事件
            if event["event"] == "on_custom":
                dispatch_custom_event(
                    name=f"data_agent_{agent_id}_{event['name']}", 
                    data={**event.get('data', {}), "agent_id": agent_id, "agent_name": agent.config.agent_name},
                    config=config
                )
            else:
                dispatch_custom_event(
                    name=f"data_agent_{agent_id}_{event['event']}", 
                    data={"agent_id": agent_id, "agent_name": agent.config.agent_name, "sub_node": event.get('name', 'unknown')},
                    config=config
                )
            
            agent_events.append({**event, "agent_id": agent_id, "agent_name": agent.config.agent_name})
            
            # 获取最终结果
            if event["event"] == "on_chain_end" and event.get("name") == "submit_result":
                agent_output = event.get("data", {}).get("output", {})
        
        # 处理结果
        factor = None
        if agent_output:
            factor = agent_output['result']
        return {"factor": factor, "events": agent_events} if factor else None

    async def _run_single_research_agent(self, agent_id: int, agent, trigger_time: str, factors: List, config: RunnableConfig):
        """运行单个research agent"""
        print(f"🔍 开始运行Research Agent {agent_id} ({agent.config.agent_name})...")
        
        try:
            # 构建背景信息
            background_information = agent.build_background_information(trigger_time, agent.config.belief, factors)
            agent_input = ResearchAgentInput(
                trigger_time=trigger_time,
                background_information=background_information
            )
            
            agent_events = []
            agent_output = None

            # 运行agent并收集事件
            async for event in agent.run_with_monitoring_events(agent_input, config):
                # 转发事件
                if event["event"] == "on_custom":
                    dispatch_custom_event(
                        name=f"research_agent_{agent_id}_{event['name']}", 
                        data={**event.get('data', {}), "agent_id": agent_id, "agent_name": agent.config.agent_name},
                        config=config
                    )
                else:
                    dispatch_custom_event(
                        name=f"research_agent_{agent_id}_{event['event']}", 
                        data={"agent_id": agent_id, "agent_name": agent.config.agent_name, "sub_node": event.get('name', 'unknown')},
                        config=config
                    )
                
                agent_events.append({**event, "agent_id": agent_id, "agent_name": agent.config.agent_name})
                
                # 获取最终结果
                if event["event"] == "on_chain_end" and event.get("name") == "submit_result":
                    agent_output = event.get("data", {}).get("output", {})
            
            # 处理结果 - 解析多个信号
            signals = []
            if agent_output:
                if "result" in agent_output and agent_output["result"]:
                    result_obj = agent_output["result"]
                    signals = self._parse_multiple_results(result_obj.final_result_thinking, result_obj.final_result)
                else:
                    signals = self._parse_multiple_results(agent_output.get("final_result_thinking", ""), agent_output.get("final_result", ""))
                
                # 为每个信号添加agent信息，最多取5个信号
                valid_signals = []
                for i, signal in enumerate(signals[:5]):
                    if signal:
                        signal["agent_id"] = agent_id
                        signal["agent_name"] = agent.config.agent_name
                        signal["signal_index"] = i + 1
                        valid_signals.append(signal)
                signals = valid_signals
            
            print(f"✅ Research Agent {agent_id} 完成，生成 {len(signals)} 个信号")
            return {"signals": signals, "events": agent_events} if signals else None
            
        except Exception as e:
            print(f"❌ Research Agent {agent_id} ({agent.config.agent_name}) 运行失败: {str(e)}")
            print(f"   错误类型: {type(e).__name__}")
            import traceback
            print(f"   详细错误信息: {traceback.format_exc()}")
            return None

    def _parse_multiple_results(self, thinking_result: str, output_result: str):
        """解析多个信号结果"""
        thinking = thinking_result.split("<Output>")[0].strip('\n').strip()
        output = output_result.split("<Output>")[-1].strip('\n').strip()
        
        signals = []
        try:
            # 查找所有signal块
            signal_blocks = re.findall(r'<signal>(.*?)</signal>', output, flags=re.DOTALL)
            
            for signal_block in signal_blocks:
                try:
                    signal = self._parse_single_signal_block(signal_block, thinking)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    print(f"Error parsing individual signal: {e}")
                    continue
        
        except Exception as e:
            print(f"Error parsing multiple results: {e}")
        
        return signals

    def _parse_single_signal_block(self, signal_block: str, thinking: str):
        """解析单个信号块"""
        try:
            has_opportunity = re.search(r"<has_opportunity>(.*?)</has_opportunity>", signal_block, flags=re.DOTALL).group(1).strip()
            action = re.search(r"<action>(.*?)</action>", signal_block, flags=re.DOTALL).group(1).strip()
            symbol_code = re.search(r"<symbol_code>(.*?)</symbol_code>", signal_block, flags=re.DOTALL).group(1).strip()
            symbol_name = re.search(r"<symbol_name>(.*?)</symbol_name>", signal_block, flags=re.DOTALL).group(1).strip()
            
            # 解析evidence_list
            evidence_list_str = re.search(r"<evidence_list>(.*?)</evidence_list>", signal_block, flags=re.DOTALL).group(1)
            evidence_list = []
            for item in evidence_list_str.split("<evidence>"):
                if '</evidence>' not in item:
                    continue
                evidence_description = item.split("</evidence>")[0].strip()
                try:
                    evidence_time = re.search(r"<time>(.*?)</time>", item, flags=re.DOTALL).group(1).strip()
                except:
                    evidence_time = "N/A"
                try:
                    evidence_from_source = re.search(r"<from_source>(.*?)</from_source>", item, flags=re.DOTALL).group(1).strip()
                except:
                    evidence_from_source = "N/A"
                    
                evidence_list.append({
                    "description": evidence_description,
                    "time": evidence_time,
                    "from_source": evidence_from_source,
                })

            # 解析limitations
            limitations_str = re.search(r"<limitations>(.*?)</limitations>", signal_block, flags=re.DOTALL).group(1)
            limitations = re.findall(r"<limitation>(.*?)</limitation>", limitations_str, flags=re.DOTALL)
            limitations = [l.strip() for l in limitations]
            
            # 解析probability
            probability = re.search(r"<probability>(.*?)</probability>", signal_block, flags=re.DOTALL).group(1).strip()
            
            # 修正symbol信息（使用配置中的首个目标市场）
            primary_market = GLOBAL_MARKET_MANAGER.config.target_markets[0] if GLOBAL_MARKET_MANAGER.config.target_markets else "CN-Stock"
            symbol_name, symbol_code = GLOBAL_MARKET_MANAGER.fix_symbol_code(primary_market, symbol_name, symbol_code)
            
            return {
                "thinking": thinking,
                "has_opportunity": has_opportunity,
                "action": action,   
                "symbol_code": symbol_code,
                "symbol_name": symbol_name,
                "evidence_list": evidence_list,
                "limitations": limitations,
                "probability": probability,
            }
        except Exception as e:
            print(f"Error parsing single signal block: {e}")
            return None

    # LangGraph工作流创建
    def create_company_workflow(self):
        """创建公司工作流"""
        workflow = StateGraph(CompanyState)
        
        # 添加节点
        workflow.add_node("run_data_agents", self.run_data_agents_step)
        workflow.add_node("run_research_agents", self.run_research_agents_step)
        workflow.add_node("run_judger_critic", self.run_judger_critic_step)
        workflow.add_node("run_contest", self.run_contest_step)
        workflow.add_node("finalize", self.finalize_step)
        
        # 设置入口点
        workflow.set_entry_point("run_data_agents")
        
        # 定义边
        workflow.add_edge("run_data_agents", "run_research_agents")
        workflow.add_edge("run_research_agents", "run_judger_critic")
        workflow.add_edge("run_judger_critic", "run_contest")
        workflow.add_edge("run_contest", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()

    async def run_company(self, trigger_time: str, config: RunnableConfig = None):
        """运行整个公司流程"""
        print("🚀 开始运行Simplified TradeCompany...")
        
        if config is None:
            config = RunnableConfig(recursion_limit=50)
        
        # 创建初始状态
        initial_state = CompanyState(
            trigger_time=trigger_time,
            data_factors=[],
            research_signals=[],
            judger_scores={},
            optimized_weights={},
            all_events=[],
            step_results={}
        )
        
        # 运行工作流
        workflow = self.create_company_workflow()
        final_state = await workflow.ainvoke(initial_state, config=config)
        
        print("✅ Simplified TradeCompany完成")
        print(f"📊 最终结果:")
        
        # 从step_results中获取更准确的统计信息
        step_results = final_state.get('step_results', {})
        data_team_results = step_results.get("data_team", {})
        research_team_results = step_results.get("research_team", {})
        
        data_factors_count = data_team_results.get("factors_count", len(final_state.get('data_factors', [])))
        research_signals_count = research_team_results.get("signals_count", len(final_state.get('research_signals', [])))
        total_events_count = len(final_state.get('all_events', []))
        
        print(f"   数据因子: {data_factors_count}")
        print(f"   研究信号: {research_signals_count}")
        print(f"   总事件: {total_events_count}")
        
        return final_state

    async def run_company_with_events(self, trigger_time: str, config: RunnableConfig = None):
        """使用事件流运行公司"""
        if config is None:
            config = RunnableConfig(recursion_limit=50)
        
        # 创建初始状态
        initial_state = CompanyState(
            trigger_time=trigger_time,
            data_factors=[],
            research_signals=[],
            judger_scores={},
            optimized_weights={},
            all_events=[],
            step_results={}
        )
        
        # 运行工作流并返回事件流
        workflow = self.create_company_workflow()
        async for event in workflow.astream_events(initial_state, version="v2", config=config):
            yield event

if __name__ == "__main__":
    async def main():
        company = SimpleTradeCompany()
        
        # 使用事件流运行
        print("🚀 开始测试Simplified TradeCompany事件流...")
        print("=" * 60)
        
        company_events = []
        final_state = None
        
        async for event in company.run_company_with_events("2025-08-12 09:00:00"):
            company_events.append(event)
            
            # 监听并打印事件
            event_type = event.get("event", "unknown")
            event_name = event.get("name", "unknown")
            
            if event_type == "on_chain_start":
                if event_name != "__start__":
                    print(f"🔄 Company开始: {event_name}")
            elif event_type == "on_chain_end":
                if event_name != "__start__":
                    print(f"✅ Company完成: {event_name}")
                    if event_name == "finalize":
                        final_state = event.get("data", {}).get("output", {})
            elif event_type == "on_custom":
                custom_name = event.get("name", "")
                custom_data = event.get("data", {})
                
                if custom_name.startswith("data_agent_"):
                    agent_id = custom_data.get("agent_id", "unknown")
                    print(f"📊 Data Agent {agent_id}: {custom_name}")
                elif custom_name.startswith("research_agent_"):
                    agent_id = custom_data.get("agent_id", "unknown")
                    print(f"🔍 Research Agent {agent_id}: {custom_name}")
                else:
                    print(f"🎯 自定义事件: {custom_name}")
        
        print("=" * 60)
        print(f"✅ 公司工作流完成:")
        if final_state:
            step_results = final_state.get('step_results', {})
            
            # 从step_results中获取统计信息
            data_team_results = step_results.get("data_team", {})
            research_team_results = step_results.get("research_team", {})
            
            data_factors_count = data_team_results.get("factors_count", len(final_state.get('data_factors', [])))
            research_signals_count = research_team_results.get("signals_count", len(final_state.get('research_signals', [])))
            total_events_count = len(final_state.get('all_events', []))
            
            print(f"   数据因子: {data_factors_count}")
            print(f"   研究信号: {research_signals_count}")
            print(f"   总事件: {total_events_count}")
        else:
            print(f"   无最终状态数据")
        print(f"   公司事件总数: {len(company_events)}")
        
        # 显示有效信号和调整权重
        if final_state:
            step_results = final_state.get('step_results', {})
            judger_result = step_results.get("judger_critic", {})
            
            print(f"\n🎯 有效信号和调整权重:")
            
            # 从JudgerCritic结果中获取权重信息
            optimized_weights = judger_result.get('optimized_weights', {})
            
            if optimized_weights:
                # 使用子脚本函数获取信号详细信息和格式化输出
                
                trigger_time = final_state.get('trigger_time', '2025-07-31 09:00:00')
                signal_names = list(optimized_weights.keys())
                
                # 使用正确的workspace目录
                signal_details = get_signal_details(trigger_time, "/Users/rayjiang/Desktop/Stepfun/trade-agent/trade_agent", signal_names)
                output_lines = format_signal_output(optimized_weights, signal_details)
                
                for line in output_lines:
                    print(line)
            else:
                print("   未找到权重优化结果")
        else:
            print("   无最终状态数据")
    
    asyncio.run(main())