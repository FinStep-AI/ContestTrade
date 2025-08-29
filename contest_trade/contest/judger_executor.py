"""
JudgerCritic 执行器
负责协调和执行所有Judger相关的功能
"""
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

from contest.judger_critic import SignalJudger, WeightOptimizer, JudgerCritic
from contest.judger_data_converter import DataFormatConverter


async def run_judger_critic_pipeline(trigger_time: str, workspace_dir: str, research_agents: Optional[List] = None) -> Dict[str, Any]:
    """
    运行完整的JudgerCritic流程
    
    Args:
        trigger_time: 触发时间
        workspace_dir: 工作目录
        research_agents: 研究代理列表（用于补全缺失信号）
        
    Returns:
        Dict: 包含评分结果、权重优化等的完整结果
    """
    print(f"🤖 开始运行JudgerCritic流程，时间: {trigger_time}")
    
    try:
        # 初始化组件
        judger_critic = JudgerCritic(workspace_dir)
        
        # 运行完整流程
        result = await judger_critic.run_judger_critic(trigger_time, research_agents)
        
        if result['status'] == 'success':
            print("✅ JudgerCritic流程完成")
            print(f"   共识评分数量: {len(result.get('consensus_scores', {}))}")
            print(f"   优化权重数量: {len(result.get('optimized_weights', {}))}")
        else:
            print(f"❌ JudgerCritic流程失败: {result.get('reason', 'unknown')}")
            
        return result
        
    except Exception as e:
        print(f"❌ JudgerCritic流程异常: {e}")
        return {
            'status': 'failed',
            'reason': str(e),
            'trigger_time': trigger_time
        }


def filter_valid_signals(signals_data: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    过滤有效信号（has_opportunity=yes）
    
    Args:
        signals_data: 信号数据字典
        
    Returns:
        Dict: 过滤后的有效信号
    """
    valid_signals = {}
    
    for signal_name, signal_data in signals_data.items():
        has_opportunity = signal_data.get('has_opportunity', 'no')
        if has_opportunity.lower() == 'yes':
            valid_signals[signal_name] = signal_data
            print(f"   ✅ 保留有效信号: {signal_name} (has_opportunity={has_opportunity})")
        else:
            print(f"   ❌ 过滤无效信号: {signal_name} (has_opportunity={has_opportunity})")
    
    return valid_signals


def calculate_signal_scores(trigger_time: str, workspace_dir: str) -> Dict[str, float]:
    """
    计算信号共识评分
    
    Args:
        trigger_time: 触发时间
        workspace_dir: 工作目录
        
    Returns:
        Dict: 共识评分字典
    """
    try:
        signal_judger = SignalJudger(workspace_dir)
        
        # 运行信号评分
        all_scores, _ = asyncio.run(signal_judger.judge_signals(trigger_time))
        
        # 计算共识评分
        weight_optimizer = WeightOptimizer(workspace_dir)
        consensus_scores = weight_optimizer.calculate_consensus_scores(all_scores)
        
        return consensus_scores
        
    except Exception as e:
        print(f"计算信号评分失败: {e}")
        return {}


def optimize_signal_weights(consensus_scores: Dict[str, float], 
                          expected_sharpe_ratios: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    优化信号权重
    
    Args:
        consensus_scores: 共识评分
        expected_sharpe_ratios: 预期夏普比率
        
    Returns:
        Dict: 优化后的权重
    """
    try:
        weight_optimizer = WeightOptimizer(".")
        
        return weight_optimizer.optimize_weights(consensus_scores, expected_sharpe_ratios)
        
    except Exception as e:
        print(f"权重优化失败: {e}")
        return {}


def get_signal_details(trigger_time: str, workspace_dir: str, signal_names: List[str]) -> Dict[str, Dict]:
    """
    获取信号详细信息
    
    Args:
        trigger_time: 触发时间
        workspace_dir: 工作目录  
        signal_names: 信号名称列表
        
    Returns:
        Dict: 信号详细信息
    """
    signal_details = {}
    
    try:
        import json
        import re
        from pathlib import Path
        
        workspace_path = Path(workspace_dir)
        reports_dir = workspace_path / "agents_workspace" / "reports"
        
        # 正确转换时间戳格式: "2025-07-31 09:00:00" -> "2025-07-31_09:00:00" 
        timestamp = trigger_time.replace(' ', '_')
        
        for signal_name in signal_names:
            try:
                report_file = reports_dir / signal_name / f"{timestamp}.json"
                if report_file.exists():
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                        final_result = report_data.get('final_result', '')
                        
                        # 解析symbol_name和action
                        symbol_name_match = re.search(r'<symbol_name>(.*?)</symbol_name>', final_result, re.DOTALL)
                        action_match = re.search(r'<action>(.*?)</action>', final_result, re.DOTALL)
                        
                        symbol_name = symbol_name_match.group(1).strip() if symbol_name_match else 'N/A'
                        action = action_match.group(1).strip() if action_match else 'N/A'
                        
                        signal_details[signal_name] = {
                            'symbol_name': symbol_name,
                            'action': action
                        }
                        
            except Exception as e:
                signal_details[signal_name] = {'symbol_name': 'N/A', 'action': 'N/A'}
        
    except Exception as e:
        print(f"获取信号详细信息失败: {e}")
    
    return signal_details


def format_signal_output(optimized_weights: Dict[str, float], 
                        signal_details: Dict[str, Dict]) -> List[str]:
    """
    格式化信号输出
    
    Args:
        optimized_weights: 优化权重
        signal_details: 信号详细信息
        
    Returns:
        List: 格式化的输出行
    """
    output_lines = []
    
    # 按权重降序排列
    sorted_weights = sorted(optimized_weights.items(), key=lambda x: x[1], reverse=True)
    
    valid_signals_count = 0
    for signal_name, weight in sorted_weights:
        if weight > 0:  # 只显示权重大于0的信号
            valid_signals_count += 1
            details = signal_details.get(signal_name, {'symbol_name': 'N/A', 'action': 'N/A'})
            symbol_name = details['symbol_name']
            action = details['action']
            output_lines.append(f"   {valid_signals_count}. {symbol_name} - {action} - 权重: {weight:.1%}")
    
    if valid_signals_count == 0:
        output_lines.append("   📊 暂无有效信号")
    
    return output_lines
