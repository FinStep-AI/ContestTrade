
"""
信号评分器
"""
import json
import re
import time
import concurrent.futures
import requests
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from config.config import cfg
from .judger_data_converter import DataFormatConverter

class SignalJudger:
    """信号评分器 - 使用多个LLM对信号进行评分"""
    
    def __init__(self, workspace_dir: str, window_m: int = 5):
        self.workspace_dir = Path(workspace_dir)
        self.judger_scores_dir = self.workspace_dir / "judger_scores"
        self.window_m = window_m
        
        # 从配置中获取judger设置
        self.contest_config = cfg.researcher_contest_config
        self.num_judgers = self.contest_config.get('num_judgers', 5)
        self.judger_config_name = self.contest_config.get('judger_config', 'llm')
        
        # 获取LLM配置
        self.llm_config = getattr(cfg, self.judger_config_name)
        
        # 创建输出目录
        self.judger_scores_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据转换器
        self.data_converter = DataFormatConverter(workspace_dir)
    
    def build_scoring_prompt(self, signals: Dict[str, Dict], historical_returns: Optional[Dict[str, float]] = None) -> str:
        """
        构建LLM批量批评提示词 - 完全对齐原脚本逻辑
        
        Args:
            signals: 所有信号数据字典 {signal_name: signal_data}
            historical_returns: 历史收益率数据
        Returns:
            str: 提示词
        """
        date = list(signals.values())[0].get('date', 'unknown')
        
        # 构建所有信号的信息
        signals_info = []
        for signal_name, signal_data in signals.items():
            # 获取历史收益率信息
            historical_info = ""
            if historical_returns and signal_name in historical_returns:
                returns = historical_returns[signal_name]
                if returns is not None:
                    historical_info = f"Average daily return over past {self.window_m} days: {returns:.2f}%"
                else:
                    historical_info = f"Average daily return over past {self.window_m} days: Insufficient data"
            else:
                historical_info = f"Average daily return over past {self.window_m} days: Insufficient data"
            
            # 获取信号详细信息
            thinking = signal_data.get('thinking', 'None')
            has_opportunity = signal_data.get('has_opportunity', 'None')
            evidence_list = signal_data.get('evidence_list', [])
            limitations = signal_data.get('limitations', 'None')
            probability = signal_data.get('probability', 'None')
            action = signal_data.get('action', 'None')
            
            # 格式化evidence_list
            evidence_text = ""
            if isinstance(evidence_list, list) and evidence_list:
                evidence_items = []
                for item in evidence_list:
                    if isinstance(item, dict):
                        # 如果是字典格式，提取description
                        description = item.get('description', '')
                        if description:
                            evidence_items.append(description)
                    elif isinstance(item, str):
                        # 如果是字符串格式，直接使用
                        if item:
                            evidence_items.append(item)
                
                if evidence_items:
                    evidence_text = "\n".join([f"- {item}" for item in evidence_items])
                else:
                    evidence_text = "None"
            else:
                evidence_text = "None"
            
            signal_info = f"""
Researcher ID: {signal_name}
Historical Performance: {historical_info}
Recommended Action: {action}
Thinking Process: {thinking}
Opportunity Assessment: {has_opportunity}
Evidence List: {evidence_text}
Limitations: {limitations}
Probability Assessment: {probability}
"""
            signals_info.append(signal_info)
        
        all_signals_text = "\n".join(signals_info)
        
        prompt = f"""
You are a strict stock investment analyst who needs to critically evaluate trading signals.

Evaluation Date: {date}

Below is the signal information from all researchers:

{all_signals_text}

Please evaluate all signals according to the following criticism criteria:

Criticism Criteria (Start from 100 points, only deduct points, no bonus points):
1. Historical Performance Issues: Poor performance over the past {self.window_m} days
2. Analysis Quality Issues: Confused thinking process, lack of depth, unclear logic
3. Insufficient Evidence Issues: Few evidence, poor quality, lack of persuasiveness, insufficient evidence
4. Risk Assessment Issues: Insufficient awareness of limitations, unreasonable probability assessment, weak risk awareness
5. Opportunity Judgment Issues: Inaccurate has_opportunity judgment, poor opportunity identification ability
6. Logical Flaws: Logical contradictions in analysis, imprecise reasoning
7. Data Issues: Improper data usage, data interpretation errors

Please output strictly according to the following format, one researcher per line:
agent_0: 75|Average historical performance(-15), insufficient analysis depth(-10), moderate evidence(-5)
...
agent_n: 45|Poor historical performance(-25), confused analysis logic(-15), insufficient evidence(-10), missing risk assessment(-5)
Format Instructions:
- Each line format: Researcher ID: Final Score|Criticism Reasons (only deduction items)
- Final score range: 0 to 100 (deduct from 100 points)
- Only question signals and logic and deduct points, no bonus points
- Criticism reasons should detail the reasons for deduction and specific problems
- Must use "|" to separate score and reasons, do not use other separators
"""
        return prompt
    
    def call_llm_for_scoring(self, prompt: str, judger_id: int, max_retries: int = 3) -> str:
        """调用LLM进行评分"""
        headers = {
            'Authorization': f'Bearer {self.llm_config["api_key"]}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.llm_config["model_name"],
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 10000,
            'temperature': 0.1
        }
        
        for attempt in range(max_retries + 1):
            try:
                print(f"    调用judger_{judger_id} ({self.llm_config['model_name']}) (尝试 {attempt + 1}/{max_retries + 1})...")
                
                response = requests.post(
                    f"{self.llm_config['base_url']}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=180
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        choice = result['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            return choice['message']['content']
                        else:
                            print(f"    警告: judger_{judger_id} 响应格式异常")
                            return f"错误: 无法解析响应内容"
                    else:
                        print(f"    错误: judger_{judger_id} API响应格式错误")
                        return f"错误: API响应格式错误"
                else:
                    print(f"    错误: judger_{judger_id} HTTP {response.status_code}: {response.text}")
                    if attempt < max_retries:
                        time.sleep(2)
                        continue
                    else:
                        return f"错误: HTTP {response.status_code}"
                        
            except Exception as e:
                print(f"    错误: judger_{judger_id} 调用失败: {e}")
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                else:
                    return f"错误: {e}"
        
        return f"错误: 经过{max_retries + 1}次尝试后仍然失败"
    
    def parse_llm_scores(self, content: str) -> Dict[str, Dict]:
        """解析LLM返回的评分结果"""
        scores = {}
        try:
            lines = content.strip().split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        signal_name = parts[0].strip()
                        score_reason_text = parts[1].strip()
                        
                        # 分离分数和理由
                        if '|' in score_reason_text:
                            score_text, reason = score_reason_text.split('|', 1)
                            reason = reason.strip()
                        elif ' - ' in score_reason_text:
                            score_text, reason = score_reason_text.split(' - ', 1)
                            reason = reason.strip()
                        else:
                            score_text = score_reason_text
                            reason = "无评分理由"
                        
                        # 提取数字
                        numbers = re.findall(r'\d+', score_text)
                        if numbers:
                            score = float(numbers[0])
                            scores[signal_name] = {
                                'score': min(max(score, 0), 100),
                                'reason': reason
                            }
        except Exception as e:
            print(f"解析评分结果出错: {e}")
        
        return scores
    
    def check_missing_signals(self, trigger_time: str, window_m: int = 5) -> List[str]:
        """
        检查过去window_m天是否有缺失的信号
        
        Args:
            trigger_time: 当前触发时间
            window_m: 历史窗口天数
            
        Returns:
            List[str]: 缺失信号的日期列表
        """
        missing_dates = []
        
        # 解析当前时间
        current_date = datetime.strptime(trigger_time, "%Y-%m-%d %H:%M:%S")
        
        # 检查过去window_m天
        for i in range(1, window_m + 1):
            check_date = current_date - timedelta(days=i)
            check_time = check_date.strftime("%Y-%m-%d %H:%M:%S")
            
            # 检查是否有信号文件
            signals = self.data_converter.load_research_signals(check_time)
            if not signals:
                missing_dates.append(check_time)
        
        return missing_dates
    
    async def run_missing_signals(self, missing_dates: List[str], research_agents) -> bool:
        """
        运行缺失的信号（不进行contest）
        
        Args:
            missing_dates: 缺失信号的日期列表
            research_agents: research agents实例
            
        Returns:
            bool: 是否成功运行
        """
        if not missing_dates:
            return True
        
        print(f"发现 {len(missing_dates)} 个缺失信号，开始补全...")
        
        for missing_time in missing_dates:
            print(f"补全时间: {missing_time}")
            try:
                # 运行research agents生成信号，但不进行contest
                # 这里需要调用research agents的run方法，但跳过contest步骤
                success = await self._run_research_agents_for_missing_signal(missing_time, research_agents)
                if success:
                    print(f"  ✅ 补全完成: {missing_time}")
                else:
                    print(f"  ❌ 补全失败: {missing_time}")
                    return False
            except Exception as e:
                print(f"  ❌ 补全失败: {missing_time} - {e}")
                return False
        
        return True
    
    async def _run_research_agents_for_missing_signal(self, trigger_time: str, research_agents) -> bool:
        """
        为缺失信号运行research agents（不进行contest）
        
        Args:
            trigger_time: 触发时间
            research_agents: research agents实例
            
        Returns:
            bool: 是否成功运行
        """
        try:
            # 这里需要实现具体的research agents运行逻辑
            # 由于research agents的运行逻辑比较复杂，这里提供一个框架
            
            # 1. 加载因子数据
            factors = self.data_converter.load_factor_data(trigger_time)
            
            # 2. 运行每个research agent
            for agent_id, agent in research_agents.items():
                try:
                    print(f"    运行agent_{agent_id}...")
                    
                    # 构建背景信息
                    background_information = await self._build_background_information(trigger_time, agent.config.belief, factors)
                    
                    # 创建agent输入
                    from agents.research_agent import ResearchAgentInput
                    agent_input = ResearchAgentInput(
                        trigger_time=trigger_time,
                        background_information=background_information,
                        output_format=self._get_output_format(),
                        task=self._get_invest_prompt(),
                    )
                    
                    # 运行agent（不进行contest）
                    agent_events = []
                    async for event in agent.run_with_monitoring_events(agent_input, config=None):
                        agent_events.append(event)
                    
                    print(f"    agent_{agent_id} 运行完成")
                    
                except Exception as e:
                    print(f"    agent_{agent_id} 运行失败: {e}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"运行research agents失败: {e}")
            return False
    
    async def _build_background_information(self, trigger_time: str, belief: str, factors: List):
        """构建背景信息"""
        import textwrap
        
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
        
        return textwrap.dedent(f"""
        <market_information>
        {global_market_information}
        </market_information>

        <your_belief>
        {belief}
        </your_belief>
        """)

    def _get_invest_prompt(self):
        """获取投资提示"""
        return """
As an professional researcher with specific belief, your need to find the opportunity in the market today. And your need to submit a critical analysis suggestion to the investor.
Your submission should should include following parts:
1. Does valuable opportunity exist in the market today?
2. Symbol Information of the opportunity
3. Evidence list your find to prove the opportunity is valuable. Judger will use these evidences to judge the opportunity is valuable or not.
4. Based on the evidence_list, your need to give a probability to this opportunity.
5. Your need to give a limitation to your suggestion, such as risk, etc. No limitation will be rejected.
7. Your can only give one opportunity suggestion.
8. If accepted, your suggestion will execute when the market open and hold for one day. So you need to focus on short-term informations.
"""

    def _get_output_format(self):
        """获取输出格式"""
        return """
<suggestion>
<has_opportunity>xxx</has_opportunity>  # yes or no
<action>xxx</action>  # buy or sell
<symbol_code>xxx</symbol_code>     # such as 600519.SH or TSLA
<symbol_name>xxx</symbol_name>  # such as Apple Inc or Tesla
<evidence_list>        # no more than 20 evidences
<evidence>xxx</evidence>   # a detailed evidence description, including convincing logical inferences which support your suggestion. About 100 words.
<time>xxx</time>           # evidence time
<from_source>xxx</from_source>   # evidence source, from which media name or website name or tools name
...
</evidence_list>
<limitations>
<limitation>xxx</limitation>   # limitations of your suggestion, such as risk, etc.
...
</limitations>
<probability>xxx</probability>  # 0-100
</suggestion>
"""
    
    def calculate_historical_returns(self, trigger_time: str) -> Optional[Dict[str, Optional[float]]]:
        """
        计算历史收益率
        
        Args:
            trigger_time: 当前触发时间
            
        Returns:
            Dict[signal_name, avg_return]: 历史平均收益率字典，None表示数据不足
        """
        try:
            from utils.market_manager import MarketManager, MarketManagerConfig
            
            # 初始化市场管理器
            market_config = MarketManagerConfig.from_config_file()
            market_manager = MarketManager(market_config)
            
            # 解析当前时间
            current_date = datetime.strptime(trigger_time, "%Y-%m-%d %H:%M:%S")
            
            # 获取所有agent的历史收益
            historical_returns = {}
            
            # 使用正确的路径访问reports目录
            reports_dir = self.workspace_dir / "reports"
            if reports_dir.exists():
                for agent_dir in reports_dir.iterdir():
                    if agent_dir.is_dir() and agent_dir.name.startswith('agent_'):
                        agent_name = agent_dir.name
                        returns = []
                        
                        # 获取过去window_m天的信号（从昨天开始，不包含今天）
                        for i in range(1, self.window_m + 1):
                            check_date = current_date - timedelta(days=i)
                            check_time = check_date.strftime("%Y-%m-%d %H:%M:%S")
                            
                            # 加载信号数据
                            signal_file = agent_dir / f"{check_time.replace(' ', '_')}.json"
                            if signal_file.exists():
                                try:
                                    with open(signal_file, 'r', encoding='utf-8') as f:
                                        signal_data = json.load(f)
                                    
                                    # 解析信号
                                    parsed_signal = self.data_converter._parse_final_result(signal_data.get('final_result', ''))
                                    if parsed_signal and parsed_signal.get('action') in ['buy', 'sell']:
                                        # 计算收益率
                                        return_value = self._calculate_signal_return(
                                            parsed_signal, check_time, market_manager
                                        )
                                        if return_value is not None:
                                            returns.append(return_value)
                                except Exception as e:
                                    print(f"计算历史收益失败 {agent_name} {check_time}: {e}")
                        
                        # 计算平均收益率
                        if returns:
                            historical_returns[agent_name] = np.mean(returns)
                        else:
                            historical_returns[agent_name] = 0.0  # 改为0.0而不是None
            
            return historical_returns if historical_returns else None
            
        except Exception as e:
            print(f"历史收益计算失败: {e}")
            return None
    
    def _calculate_signal_return(self, signal_data: Dict, signal_time: str, market_manager) -> Optional[float]:
        """
        计算信号的过去五个交易日收益率（基于开盘价）
        
        对于buy信号：计算过去5个交易日的正向收益率
        对于sell信号：计算过去5个交易日的反向收益率（股价下跌对应正收益）
        
        Args:
            signal_data: 信号数据
            signal_time: 信号时间  
            market_manager: 市场管理器
            
        Returns:
            float: 过去五个交易日的累计收益率，None表示无法计算
        """
        try:
            action = signal_data.get('action', '')
            symbol_code = signal_data.get('symbol_code', '')
            
            if not action or not symbol_code:
                return None
            
            # 获取过去5个交易日的价格数据（需要6个点：T-5到T0）
            open_prices = []
            primary_market = market_manager.config.target_markets[0] if getattr(market_manager, 'config', None) and market_manager.config.target_markets else "CN-Stock"
            for i in range(6):  # 需要6个数据点来计算5个交易日的收益率
                try:
                    price_data = market_manager.get_symbol_price(primary_market, symbol_code, signal_time, -i)
                    if not price_data:
                        break
                    
                    open_price = price_data.get('open')
                    if open_price is not None and open_price > 0:
                        open_prices.append(open_price)
                    else:
                        break
                except:
                    break
            
            # 需要至少6个价格点来计算5个交易日收益率
            if len(open_prices) < 6:
                # 如果数据不足，尝试计算可用天数的收益率
                if len(open_prices) >= 2:
                    start_price = open_prices[-1]  # 最早的开盘价
                    end_price = open_prices[0]     # 当前日的开盘价
                    
                    # 计算基础收益率
                    base_return = (end_price - start_price) / start_price
                    
                    # 根据action调整收益率
                    if action.lower() == 'buy':
                        # buy信号：股价上涨为正收益
                        final_return = base_return
                    elif action.lower() == 'sell':
                        # sell信号：股价下跌为正收益，所以取负值
                        final_return = -base_return
                    else:
                        return None
                    
                    # 限制收益率在合理范围内
                    return max(-1.0, min(1.0, final_return))
                else:
                    return None
            
            # 计算完整5个交易日的累计收益率
            start_price = open_prices[5]  # 5个交易日前的开盘价
            end_price = open_prices[0]    # 当前日的开盘价
            
            # 计算基础收益率
            base_return = (end_price - start_price) / start_price
            
            # 根据action调整收益率
            if action.lower() == 'buy':
                # buy信号：股价上涨为正收益
                final_return = base_return
            elif action.lower() == 'sell':
                # sell信号：股价下跌为正收益，所以取负值
                final_return = -base_return
            else:
                return None
            
            # 限制收益率在合理范围内（5日累计收益率限制在±100%）
            final_return = max(-1.0, min(1.0, final_return))
            
            return final_return
            
        except Exception as e:
            print(f"计算信号过去5个交易日收益率失败: {e}")
            return None
    
    def calculate_expected_sharpe_ratios(self, trigger_time: str, window_n: int = 3) -> Optional[Dict[str, float]]:
        """
        计算预期夏普比率
        
        Args:
            trigger_time: 当前触发时间
            window_n: 未来窗口天数
            
        Returns:
            Dict[signal_name, sharpe_ratio]: 预期夏普比率字典，None表示数据不足
        """
        try:
            from utils.market_manager import MarketManager, MarketManagerConfig
            
            # 初始化市场管理器
            market_config = MarketManagerConfig.from_config_file()
            market_manager = MarketManager(market_config)
            
            # 解析当前时间
            current_date = datetime.strptime(trigger_time, "%Y-%m-%d %H:%M:%S")
            
            # 获取所有agent的预期夏普比率
            expected_sharpe_ratios = {}
            
            # 遍历所有agent目录
            if self.data_converter.reports_dir.exists():
                for agent_dir in self.data_converter.reports_dir.iterdir():
                    if agent_dir.is_dir() and agent_dir.name.startswith('agent_'):
                        agent_name = agent_dir.name
                        daily_returns = []
                        
                        # 获取未来window_n天的信号（只考虑buy信号）
                        for i in range(window_n):
                            future_date = current_date + timedelta(days=i)
                            future_time = future_date.strftime("%Y-%m-%d %H:%M:%S")
                            
                            # 加载信号数据
                            signal_file = agent_dir / f"{future_time.replace(' ', '_')}.json"
                            if signal_file.exists():
                                try:
                                    with open(signal_file, 'r', encoding='utf-8') as f:
                                        signal_data = json.load(f)
                                    
                                    # 解析信号
                                    parsed_signal = self.data_converter._parse_final_result(signal_data.get('final_result', ''))
                                    if parsed_signal and parsed_signal.get('action') == 'buy':
                                        # 计算收益率
                                        return_value = self._calculate_signal_return(
                                            parsed_signal, future_time, market_manager
                                        )
                                        if return_value is not None:
                                            daily_returns.append(return_value)
                                except Exception as e:
                                    print(f"计算预期夏普失败 {agent_name} {future_time}: {e}")
                        
                        # 计算夏普比率
                        if len(daily_returns) > 1:
                            mean_return = np.mean(daily_returns)
                            std_return = np.std(daily_returns)
                            if std_return > 0:
                                # 年化夏普比率（假设252个交易日）
                                sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
                                expected_sharpe_ratios[agent_name] = sharpe_ratio
                            else:
                                expected_sharpe_ratios[agent_name] = 0.0
                        elif len(daily_returns) == 1:
                            expected_sharpe_ratios[agent_name] = 0.0
                        else:
                            expected_sharpe_ratios[agent_name] = 0.0  # 改为0.0而不是None
            
            return expected_sharpe_ratios if expected_sharpe_ratios else None
            
        except Exception as e:
            print(f"预期夏普比率计算失败: {e}")
            return None
    
    async def judge_signals(self, trigger_time: str) -> Tuple[Dict, Dict]:
        """
        对信号进行评分
        
        Args:
            trigger_time: 触发时间
            
        Returns:
            tuple: (评分结果, 原始响应)
        """
        print(f"开始对时间 {trigger_time} 的信号进行评分...")
        
        # 加载数据
        signals = self.data_converter.load_research_signals(trigger_time)
        factors = self.data_converter.load_factor_data(trigger_time)
        
        if not signals:
            print("没有找到信号数据")
            return {}, {}
        
        print(f"加载了 {len(signals)} 个信号")
        
        # 转换数据格式
        converted_signals = self.data_converter.convert_signals_for_judging(signals, factors)
        
        if not converted_signals:
            print("信号数据转换失败")
            return {}, {}
        
        # 计算历史收益率
        historical_returns = self.calculate_historical_returns(trigger_time)
        
        # 构建prompt
        prompt = self.build_scoring_prompt(converted_signals, historical_returns)
        
        # 并发调用多个judger
        all_scores = {}
        all_responses = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_judgers) as executor:
            # 提交所有judger任务
            future_to_judger = {}
            for judger_id in range(self.num_judgers):
                future = executor.submit(self._score_with_single_judger, judger_id, prompt)
                future_to_judger[future] = judger_id
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_judger):
                judger_id = future_to_judger[future]
                try:
                    response, scores = future.result()
                    judger_name = f"judger_{judger_id}"
                    all_scores[judger_name] = scores
                    all_responses[judger_name] = response
                    print(f"  judger_{judger_id} 完成评分，解析了 {len(scores)} 个信号")
                except Exception as exc:
                    print(f"  judger_{judger_id} 评分失败: {exc}")
                    judger_name = f"judger_{judger_id}"
                    all_scores[judger_name] = {}
                    all_responses[judger_name] = f"评分失败: {exc}"
        
        # 保存结果
        self._save_judge_results(trigger_time, all_scores, all_responses)
        
        return all_scores, all_responses
    
    def _score_with_single_judger(self, judger_id: int, prompt: str) -> Tuple[str, Dict]:
        """单个judger评分的辅助方法"""
        response = self.call_llm_for_scoring(prompt, judger_id)
        scores = self.parse_llm_scores(response)
        return response, scores
    
    def _save_judge_results(self, trigger_time: str, all_scores: Dict, all_responses: Dict):
        """保存评分结果"""
        timestamp = trigger_time.replace(' ', '_').replace(':', '')
        
        # 保存详细评分结果
        scores_file = self.judger_scores_dir / f"judge_scores_{timestamp}.json"
        with open(scores_file, 'w', encoding='utf-8') as f:
            json.dump({
                'trigger_time': trigger_time,
                'scores': all_scores,
                'responses': all_responses
            }, f, ensure_ascii=False, indent=2)
        
        print(f"评分结果已保存到: {scores_file}")
