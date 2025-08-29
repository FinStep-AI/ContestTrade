"""
权重优化器
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import re
from datetime import datetime, timedelta

class WeightOptimizer:
    """权重优化器 - 基于共识评分和过去5天收益率的综合评分调整权重"""
    
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.judger_scores_dir = self.workspace_dir / "judger_scores"
        self.final_result_dir = self.workspace_dir / "final_result"
        self.reports_dir = self.workspace_dir / "reports"
        
        # 创建输出目录
        self.final_result_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_consensus_scores(self, all_scores: Dict[str, Dict]) -> Dict[str, float]:
        """计算共识评分"""
        consensus_scores = {}
        
        # 获取所有信号名称
        all_signals = set()
        for judger_scores in all_scores.values():
            all_signals.update(judger_scores.keys())
        
        # 计算每个信号的平均分
        for signal_name in all_signals:
            scores = []
            for judger_name, judger_scores in all_scores.items():
                if signal_name in judger_scores and 'score' in judger_scores[signal_name]:
                    scores.append(judger_scores[signal_name]['score'])
            
            if scores:
                consensus_scores[signal_name] = np.mean(scores)
            else:
                consensus_scores[signal_name] = 0.0
        
        return consensus_scores
    
    def get_signal_historical_returns(self, signal_name: str, trigger_time: str) -> Optional[float]:
        """
        获取某个agent过去5天信号执行的累计收益率
        
        Args:
            signal_name: 信号名称 (如 agent_1, agent_2)
            trigger_time: 触发时间
            
        Returns:
            float: 过去5天信号执行的累计收益率，如果无法获取则返回None
        """
        try:
            from datetime import datetime, timedelta
            import sys
            from pathlib import Path
            
            # 添加trade_agent目录到sys.path
            trade_agent_path = Path(__file__).parent.parent
            if str(trade_agent_path) not in sys.path:
                sys.path.insert(0, str(trade_agent_path))
            
            from utils.market_manager import GLOBAL_MARKET_MANAGER
            
            # 解析触发时间
            trigger_dt = datetime.strptime(trigger_time, "%Y-%m-%d %H:%M:%S")
            
            print(f"   📊 计算{signal_name}过去5天信号执行收益率...")
            
            # 获取过去5个交易日的信号数据
            daily_returns = []
            cumulative_return = 1.0  # 初始资金为1
            
            for i in range(1, 6):  # 过去5天，从-1天到-5天
                # 计算历史日期
                past_date = trigger_dt - timedelta(days=i)
                past_date_str = past_date.strftime("%Y-%m-%d %H:%M:%S")
                past_timestamp = past_date_str.replace(' ', '_')
                
                # 读取该日期的信号报告
                report_file = self.reports_dir / signal_name / f"{past_timestamp}.json"
                
                if not report_file.exists():
                    print(f"     ⚠️  缺少{past_date.strftime('%Y-%m-%d')}的信号数据")
                    continue
                
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                        final_result = report_data.get('final_result', '')
                        
                        # 解析symbol_code和action
                        symbol_code_match = re.search(r'<symbol_code>(.*?)</symbol_code>', final_result, re.DOTALL)
                        action_match = re.search(r'<action>(.*?)</action>', final_result, re.DOTALL)
                        has_opportunity_match = re.search(r'<has_opportunity>(.*?)</has_opportunity>', final_result, re.DOTALL)
                        
                        if not symbol_code_match or not action_match or not has_opportunity_match:
                            print(f"     ⚠️  {past_date.strftime('%Y-%m-%d')}信号格式错误")
                            continue
                        
                        has_opportunity = has_opportunity_match.group(1).strip().lower()
                        if has_opportunity != 'yes':
                            print(f"     📊 {past_date.strftime('%Y-%m-%d')}: 无机会信号，跳过")
                            continue
                        
                        symbol_code = symbol_code_match.group(1).strip()
                        action = action_match.group(1).strip()
                        
                        # 计算该信号的单日收益率
                        daily_return = self._calculate_signal_daily_return(
                            symbol_code, action, past_date_str, trigger_dt
                        )
                        
                        if daily_return is not None:
                            daily_returns.append(daily_return)
                            cumulative_return *= (1 + daily_return)
                            print(f"     📊 {past_date.strftime('%Y-%m-%d')}: {symbol_code} {action} -> {daily_return:.2%}")
                        
                except Exception as e:
                    print(f"     ❌ 解析{past_date.strftime('%Y-%m-%d')}信号失败: {e}")
                    continue
            
            if not daily_returns:
                print(f"   ⚠️  {signal_name}过去5天无有效信号数据")
                return None
            
            # 计算5天累计收益率
            total_return = cumulative_return - 1.0
            print(f"   📊 {signal_name}过去{len(daily_returns)}天累计收益率: {total_return:.2%}")
            
            return total_return
                
        except Exception as e:
            print(f"获取{signal_name}历史收益率失败: {e}")
            return None
    
    def _calculate_signal_daily_return(self, symbol_code: str, action: str, 
                                     signal_time: str, current_time: datetime) -> Optional[float]:
        """
        计算单个信号的日收益率（持有一天）
        
        Args:
            symbol_code: 股票代码
            action: 操作类型 (buy/sell)
            signal_time: 信号时间
            current_time: 当前时间（用于计算持有期）
            
        Returns:
            float: 该信号的日收益率
        """
        try:
            from datetime import datetime, timedelta
            import sys
            from pathlib import Path
            
            # 添加trade_agent目录到sys.path
            trade_agent_path = Path(__file__).parent.parent
            if str(trade_agent_path) not in sys.path:
                sys.path.insert(0, str(trade_agent_path))
            
            from utils.market_manager import GLOBAL_MARKET_MANAGER
            
            # 解析信号日期
            signal_dt = datetime.strptime(signal_time, "%Y-%m-%d %H:%M:%S")
            
            # 获取信号日开盘价（买入/卖出价格）
            primary_market = GLOBAL_MARKET_MANAGER.config.target_markets[0] if GLOBAL_MARKET_MANAGER.config.target_markets else "CN-Stock"
            entry_price_data = GLOBAL_MARKET_MANAGER.get_symbol_price(primary_market, symbol_code, signal_time, 0)
            if not entry_price_data or 'open' not in entry_price_data:
                return None
            entry_price = float(entry_price_data['open'])
            
            # 获取次日开盘价（平仓价格）
            next_day = signal_dt + timedelta(days=1)
            next_day_str = next_day.strftime("%Y-%m-%d %H:%M:%S")
            exit_price_data = GLOBAL_MARKET_MANAGER.get_symbol_price(primary_market, symbol_code, next_day_str, 0)
            if not exit_price_data or 'open' not in exit_price_data:
                return None
            exit_price = float(exit_price_data['open'])
            
            # 计算收益率
            if action.lower() == 'buy':
                # 买入信号：次日开盘价相对于当日开盘价的涨幅
                return (exit_price - entry_price) / entry_price
            elif action.lower() == 'sell':
                # 卖出信号：当日开盘价相对于次日开盘价的涨幅（做空收益）
                return (entry_price - exit_price) / entry_price
            else:
                return None
                
        except Exception as e:
            print(f"     ❌ 计算{symbol_code}日收益率失败: {e}")
            return None
    
    def optimize_weights(self, consensus_scores: Dict[str, float], trigger_time: str) -> Dict[str, float]:
        """
        基于共识评分和过去5天收益率优化权重
        
        Args:
            consensus_scores: 共识评分
            trigger_time: 触发时间
            
        Returns:
            Dict[signal_name, weight]: 优化后的权重
        """
        signal_names = list(consensus_scores.keys())
        
        if not signal_names:
            return {}
        
        print("🔄 正在计算综合评分...")
        
        # 计算每个信号的综合评分
        composite_scores = {}
        
        for signal_name in signal_names:
            consensus_score = consensus_scores[signal_name]
            
            # 获取历史收益率
            historical_return = self.get_signal_historical_returns(signal_name, trigger_time)
            
            if historical_return is None:
                print(f"   📊 {signal_name}: 共识评分={consensus_score:.1f}, 历史收益率=无数据 -> 综合评分=0")
                composite_scores[signal_name] = 0.0
            elif historical_return <= 0:
                print(f"   📊 {signal_name}: 共识评分={consensus_score:.1f}, 历史收益率={historical_return:.2%} (负值) -> 综合评分=0")
                composite_scores[signal_name] = 0.0
            else:
                historical_weight = 0.5 * historical_return
                composite_score = consensus_score * (1 + historical_weight)
                composite_scores[signal_name] = composite_score
                print(f"   📊 {signal_name}: 共识评分={consensus_score:.1f}, 历史收益率={historical_return:.2%} (正值) -> 综合评分={composite_score:.2f}")
        
        # 计算权重
        return self._calculate_composite_weights(composite_scores)
    
    def _calculate_composite_weights(self, composite_scores: Dict[str, float]) -> Dict[str, float]:
        """
        基于综合评分计算权重
        
        Args:
            composite_scores: 综合评分
            
        Returns:
            Dict[signal_name, weight]: 权重字典
        """
        signal_names = list(composite_scores.keys())
        scores = np.array([composite_scores[name] for name in signal_names])
        
        # 过滤掉评分为0或负数的信号
        positive_mask = scores > 0
        
        if not np.any(positive_mask):
            # 如果所有信号的综合评分都不是正数，分配0权重
            print("   ⚠️  所有信号的综合评分都不是正数，所有权重设为0")
            return {name: 0.0 for name in signal_names}
        
        # 只对正评分的信号分配权重
        positive_scores = scores[positive_mask]
        positive_names = [signal_names[i] for i in range(len(signal_names)) if positive_mask[i]]
        
        # 按评分大小加权分配
        total_score = np.sum(positive_scores)
        weights = positive_scores / total_score
        
        # 构建权重字典
        weight_dict = {}
        positive_idx = 0
        for i, signal_name in enumerate(signal_names):
            if positive_mask[i]:
                weight_dict[signal_name] = float(weights[positive_idx])
                positive_idx += 1
            else:
                weight_dict[signal_name] = 0.0
        
        return weight_dict
    
    def _calculate_score_weights(self, consensus_scores: Dict[str, float]) -> Dict[str, float]:
        """
        基于评分计算权重 - 按大小加权分配
        
        Args:
            consensus_scores: 共识评分
            
        Returns:
            Dict[signal_name, weight]: 权重字典
        """
        signal_names = list(consensus_scores.keys())
        scores = np.array([consensus_scores[name] for name in signal_names])
        
        # 避免所有分数都为0的情况
        if np.all(scores == 0):
            weights = np.ones(len(scores)) / len(scores)
        else:
            # 按评分大小加权分配（线性加权）
            total_score = np.sum(scores)
            weights = scores / total_score
        
        # 构建权重字典
        weight_dict = {}
        for i, signal_name in enumerate(signal_names):
            weight_dict[signal_name] = float(weights[i])
        
        return weight_dict
    
    def save_final_results(self, trigger_time: str, consensus_scores: Dict[str, float], 
                          optimized_weights: Dict[str, float]):
        """保存最终结果"""
        timestamp = trigger_time.replace(' ', '_').replace(':', '')
        
        # 构建最终结果
        final_result = {
            'trigger_time': trigger_time,
            'consensus_scores': consensus_scores,
            'optimized_weights': optimized_weights,
            'summary': {
                'total_signals': len(consensus_scores),
                'avg_score': np.mean(list(consensus_scores.values())) if consensus_scores else 0,
                'top_signals': sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            }
        }
        
        # 保存到文件
        result_file = self.final_result_dir / f"final_result_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        print(f"最终结果已保存到: {result_file}")
        return final_result

