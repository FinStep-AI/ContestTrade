"""
信号评估和权重优化系统 - 基于LLM评分的信号筛选和权重调整
"""
import json
import os
import asyncio
import textwrap
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import requests
import concurrent.futures
import warnings
import re
import logging
from collections import defaultdict


from config.config import cfg, PROJECT_ROOT
from contest.judger_weight_optimizer import WeightOptimizer
from agents.research_agent import ResearchAgentInput
from config.config import cfg
from models.llm_model import GLOBAL_LLM

warnings.filterwarnings('ignore')

class DataFormatConverter:
    """数据格式转换器，将新格式数据转换为评分系统所需格式"""
    
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.reports_dir = self.workspace_dir / "reports"
        self.factors_dir = self.workspace_dir / "factors"
    
    def load_research_signals(self, trigger_time: str) -> Dict[str, Dict]:
        """
        加载研究信号数据
        
        Args:
            trigger_time: 触发时间，格式为 "2025-08-07 09:00:00"
            
        Returns:
            Dict[agent_name, signal_data]: 信号数据字典
        """
        signals = {}
        
        # 生成文件名 (保留冒号，只替换空格为下划线)
        filename = f"{trigger_time.replace(' ', '_')}.json"
        
        # 遍历所有agent目录
        if self.reports_dir.exists():
            for agent_dir in self.reports_dir.iterdir():
                if agent_dir.is_dir() and agent_dir.name.startswith('agent_'):
                    signal_file = agent_dir / filename
                    if signal_file.exists():
                        try:
                            with open(signal_file, 'r', encoding='utf-8') as f:
                                signal_data = json.load(f)
                            signals[agent_dir.name] = signal_data
                        except Exception as e:
                            print(f"加载信号文件失败 {signal_file}: {e}")
        
        return signals
    
    def load_factor_data(self, trigger_time: str) -> Dict[str, Dict]:
        """
        加载因子数据
        
        Args:
            trigger_time: 触发时间
            
        Returns:
            Dict[agent_name, factor_data]: 因子数据字典
        """
        factors = {}
        
        # 生成文件名 (保留冒号，只替换空格为下划线)
        filename = f"{trigger_time.replace(' ', '_')}.json"
        
        # 遍历所有factor目录
        if self.factors_dir.exists():
            for factor_dir in self.factors_dir.iterdir():
                if factor_dir.is_dir():
                    factor_file = factor_dir / filename
                    if factor_file.exists():
                        try:
                            with open(factor_file, 'r', encoding='utf-8') as f:
                                factor_data = json.load(f)
                            factors[factor_dir.name] = factor_data
                        except Exception as e:
                            print(f"加载因子文件失败 {factor_file}: {e}")
        
        return factors
    
    def convert_signals_for_judging(self, signals: Dict[str, Dict], factors: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        将信号数据转换为评分系统所需格式
        
        Args:
            signals: 研究信号数据
            factors: 因子数据
            
        Returns:
            Dict[signal_name, signal_data]: 转换后的信号数据
        """
        converted_signals = {}
        
        for agent_name, signal_data in signals.items():
            # 解析final_result获取结构化数据
            parsed_signal = self._parse_final_result(signal_data.get('final_result', ''))
            
            if parsed_signal:
                # 构建标准化的信号数据
                signal_name = agent_name
                converted_signal = {
                    'signal_name': signal_name,
                    'date': signal_data.get('trigger_time', ''),
                    'thinking': signal_data.get('final_result_thinking', ''),
                    'has_opportunity': parsed_signal.get('has_opportunity', 'no'),
                    'action': parsed_signal.get('action', 'none'),
                    'symbol_code': parsed_signal.get('symbol_code', ''),
                    'symbol_name': parsed_signal.get('symbol_name', ''),
                    'evidence_list': parsed_signal.get('evidence_list', []),
                    'limitations': parsed_signal.get('limitations', []),
                    'probability': parsed_signal.get('probability', '0'),
                    'belief': signal_data.get('belief', ''),
                    'background_information': signal_data.get('background_information', '')
                }
                converted_signals[signal_name] = converted_signal
        
        return converted_signals
    
    def _parse_final_result(self, final_result: str) -> Optional[Dict]:
        """解析final_result字符串，提取结构化数据"""
        try:
            # 移除<Output>标签
            if '<Output>' in final_result:
                final_result = final_result.split('<Output>')[-1].strip()
            
            # 提取各个字段
            has_opportunity = self._extract_field(final_result, 'has_opportunity')
            action = self._extract_field(final_result, 'action')
            symbol_code = self._extract_field(final_result, 'symbol_code')
            symbol_name = self._extract_field(final_result, 'symbol_name')
            probability = self._extract_field(final_result, 'probability')
            
            # 提取evidence_list
            evidence_list = self._extract_evidence_list(final_result)
            
            # 提取limitations
            limitations = self._extract_limitations(final_result)
            
            return {
                'has_opportunity': has_opportunity,
                'action': action,
                'symbol_code': symbol_code,
                'symbol_name': symbol_name,
                'evidence_list': evidence_list,
                'limitations': limitations,
                'probability': probability
            }
        except Exception as e:
            print(f"解析final_result失败: {e}")
            return None
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """提取单个字段"""
        pattern = f"<{field_name}>(.*?)</{field_name}>"
        match = re.search(pattern, text, flags=re.DOTALL)
        return match.group(1).strip() if match else ''
    
    def _extract_evidence_list(self, text: str) -> List[Dict]:
        """提取evidence_list"""
        evidence_list = []
        
        # 提取整个evidence_list内容
        evidence_list_match = re.search(r"<evidence_list>(.*?)</evidence_list>", text, flags=re.DOTALL)
        if not evidence_list_match:
            return evidence_list
        
        evidence_list_content = evidence_list_match.group(1)
        
        # 分割每个evidence块
        evidence_blocks = re.split(r"<evidence>", evidence_list_content)
        
        for block in evidence_blocks:
            if '</evidence>' in block:
                evidence_parts = block.split('</evidence>')
                if len(evidence_parts) >= 1:
                    evidence_content = evidence_parts[0].strip()
                    
                    # 提取time和from_source
                    time_match = re.search(r"<time>(.*?)</time>", evidence_parts[0] if len(evidence_parts) > 1 else block, flags=re.DOTALL)
                    source_match = re.search(r"<from_source>(.*?)</from_source>", evidence_parts[0] if len(evidence_parts) > 1 else block, flags=re.DOTALL)
                    
                    evidence_list.append({
                        'description': evidence_content,
                        'time': time_match.group(1).strip() if time_match else '',
                        'from_source': source_match.group(1).strip() if source_match else ''
                    })
        
        return evidence_list
    
    def _extract_limitations(self, text: str) -> List[str]:
        """提取limitations"""
        limitations = []
        
        # 提取整个limitations内容
        limitations_match = re.search(r"<limitations>(.*?)</limitations>", text, flags=re.DOTALL)
        if not limitations_match:
            return limitations
        
        limitations_content = limitations_match.group(1)
        
        # 提取每个limitation
        limitation_matches = re.findall(r"<limitation>(.*?)</limitation>", limitations_content, flags=re.DOTALL)
        for limitation in limitation_matches:
            limitations.append(limitation.strip())
        
        return limitations


class SignalJudger:
    """信号评分器 - 使用多个LLM对信号进行评分，增强客观数据验证"""
    
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.judger_scores_dir = self.workspace_dir / "judger_scores"
        self.window_m = cfg.researcher_contest_config.get('window_m', 5)
        
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
        
        # 初始化日志
        self.logger = logging.getLogger(f"SignalJudger")
        
        # 客观验证指标配置
        self.validation_config = {
            'min_evidence_count': 2,  # 最少证据数量
            'min_data_citation_ratio': 0.6,  # 最少数据引用比例
            'max_uncertainty_threshold': 0.8,  # 最大不确定性阈值
            'required_data_sources': ['price_info', 'corp_info', 'search_web'],  # 必需数据源
            'credibility_weights': {
                'data_citation_quality': 0.3,
                'evidence_consistency': 0.25,
                'source_reliability': 0.2,
                'logical_coherence': 0.15,
                'uncertainty_handling': 0.1
            }
        }
    
    def validate_signal_data_quality(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        客观验证信号数据质量
        
        Args:
            signal_data: 信号数据
            
        Returns:
            Dict: 验证结果和质量指标
        """
        validation_result = {
            'is_valid': True,
            'quality_score': 0.0,
            'validation_details': {},
            'credibility_score': 0.0,
            'issues': []
        }
        
        # 1. 检查证据数量和质量
        evidence_list = signal_data.get('evidence_list', [])
        evidence_count = len(evidence_list)
        
        if evidence_count < self.validation_config['min_evidence_count']:
            validation_result['issues'].append(f"证据数量不足: {evidence_count} < {self.validation_config['min_evidence_count']}")
            validation_result['is_valid'] = False
        
        # 2. 检查数据引用格式和质量
        citation_quality = self._analyze_data_citations(evidence_list)
        validation_result['validation_details']['citation_quality'] = citation_quality
        
        if citation_quality['citation_ratio'] < self.validation_config['min_data_citation_ratio']:
            validation_result['issues'].append(f"数据引用比例过低: {citation_quality['citation_ratio']:.2%}")
        
        # 3. 检查不确定性处理
        uncertainty_analysis = self._analyze_uncertainty_handling(signal_data)
        validation_result['validation_details']['uncertainty_analysis'] = uncertainty_analysis
        
        # 4. 检查数据源多样性
        source_diversity = self._analyze_source_diversity(evidence_list)
        validation_result['validation_details']['source_diversity'] = source_diversity
        
        # 5. 检查逻辑一致性
        logical_coherence = self._analyze_logical_coherence(signal_data)
        validation_result['validation_details']['logical_coherence'] = logical_coherence
        
        # 6. 计算综合可信度分数
        credibility_score = self._calculate_credibility_score(
            citation_quality, uncertainty_analysis, source_diversity, logical_coherence
        )
        validation_result['credibility_score'] = credibility_score
        
        # 7. 计算总体质量分数
        quality_components = [
            min(evidence_count / self.validation_config['min_evidence_count'], 1.0),
            citation_quality['citation_ratio'],
            source_diversity['diversity_score'],
            logical_coherence['coherence_score'],
            1.0 - uncertainty_analysis['uncertainty_level']  # 不确定性越低，质量越高
        ]
        
        validation_result['quality_score'] = sum(quality_components) / len(quality_components)
        
        return validation_result
    
    def _analyze_data_citations(self, evidence_list: List[str]) -> Dict[str, Any]:
        """
        分析数据引用质量
        """
        total_evidence = len(evidence_list)
        cited_evidence = 0
        citation_formats = []
        
        # 标准引用格式模式
        citation_patterns = [
            r'\[\w+\|[\d\-\s:]+\|[^\]]+\]',  # [工具名|时间|数值]
            r'\[\w+\s*\w*\|[\d\-\s:]+\|[^\]]+\]',  # [数据源|时间|标题]
        ]
        
        for evidence in evidence_list:
            has_citation = False
            for pattern in citation_patterns:
                if re.search(pattern, evidence):
                    cited_evidence += 1
                    citation_formats.append(pattern)
                    has_citation = True
                    break
            
            if not has_citation:
                # 检查是否有其他形式的数据引用
                if any(keyword in evidence.lower() for keyword in ['根据', '数据显示', '报告显示']):
                    cited_evidence += 0.5  # 部分分数
        
        citation_ratio = cited_evidence / total_evidence if total_evidence > 0 else 0
        
        return {
            'total_evidence': total_evidence,
            'cited_evidence': cited_evidence,
            'citation_ratio': citation_ratio,
            'citation_formats_used': len(set(citation_formats))
        }
    
    def _analyze_uncertainty_handling(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析不确定性处理质量
        """
        uncertainty_indicators = {
            'probability': signal_data.get('probability', 0),
            'limitations_provided': bool(signal_data.get('limitations', '').strip()),
            'confidence_levels': []
        }
        
        # 检查证据中的确定性级别标注
        evidence_list = signal_data.get('evidence_list', [])
        for evidence in evidence_list:
            # 查找确定性级别标注 [确定性级别: XX%]
            confidence_matches = re.findall(r'\[确定性级别[：:]\s*(\d+)%\]', evidence)
            if confidence_matches:
                uncertainty_indicators['confidence_levels'].extend([int(x) for x in confidence_matches])
        
        # 计算平均确定性级别
        avg_confidence = sum(uncertainty_indicators['confidence_levels']) / len(uncertainty_indicators['confidence_levels']) if uncertainty_indicators['confidence_levels'] else 0
        
        # 计算不确定性级别 (0-1, 越低越好)
        uncertainty_level = 1.0 - (avg_confidence / 100.0) if avg_confidence > 0 else 0.5
        
        return {
            'probability': uncertainty_indicators['probability'],
            'has_limitations': uncertainty_indicators['limitations_provided'],
            'confidence_levels': uncertainty_indicators['confidence_levels'],
            'avg_confidence': avg_confidence,
            'uncertainty_level': uncertainty_level
        }
    
    def _analyze_source_diversity(self, evidence_list: List[str]) -> Dict[str, Any]:
        """
        分析数据源多样性
        """
        sources_found = set()
        
        # 从引用中提取数据源
        for evidence in evidence_list:
            # 查找引用格式中的数据源
            source_matches = re.findall(r'\[([^\|]+)\|', evidence)
            sources_found.update(source_matches)
        
        required_sources = set(self.validation_config['required_data_sources'])
        sources_coverage = len(sources_found.intersection(required_sources)) / len(required_sources)
        diversity_score = min(len(sources_found) / 3.0, 1.0)  # 最多3个不同源得满分
        
        return {
            'sources_found': list(sources_found),
            'sources_count': len(sources_found),
            'required_sources_coverage': sources_coverage,
            'diversity_score': diversity_score
        }
    
    def _analyze_logical_coherence(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析逻辑一致性
        """
        coherence_score = 1.0
        issues = []
        
        # 检查行动建议与概率的一致性
        action = signal_data.get('action', '').lower()
        probability = signal_data.get('probability', 0)
        
        if action in ['buy', 'sell'] and probability < 0.6:
            coherence_score -= 0.3
            issues.append(f"行动建议({action})与概率({probability})不一致")
        
        # 检查机会判断与行动的一致性
        has_opportunity = signal_data.get('has_opportunity', False)
        if has_opportunity and action == 'hold':
            coherence_score -= 0.2
            issues.append("发现机会但建议持有，逻辑不一致")
        
        # 检查证据与结论的一致性（基于关键词）
        evidence_text = ' '.join(signal_data.get('evidence_list', []))
        positive_keywords = ['增长', '上涨', '利好', '强劲', '超预期']
        negative_keywords = ['下跌', '下降', '利空', '疲软', '低于预期']
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in evidence_text)
        negative_count = sum(1 for keyword in negative_keywords if keyword in evidence_text)
        
        if action == 'buy' and negative_count > positive_count:
            coherence_score -= 0.2
            issues.append("买入建议与负面证据不符")
        elif action == 'sell' and positive_count > negative_count:
            coherence_score -= 0.2
            issues.append("卖出建议与正面证据不符")
        
        return {
            'coherence_score': max(coherence_score, 0.0),
            'issues': issues
        }
    
    def _calculate_credibility_score(self, citation_quality: Dict, uncertainty_analysis: Dict, 
                                   source_diversity: Dict, logical_coherence: Dict) -> float:
        """
        计算综合可信度分数
        """
        weights = self.validation_config['credibility_weights']
        
        components = {
            'data_citation_quality': citation_quality['citation_ratio'],
            'evidence_consistency': 1.0 - uncertainty_analysis['uncertainty_level'],
            'source_reliability': source_diversity['diversity_score'],
            'logical_coherence': logical_coherence['coherence_score'],
            'uncertainty_handling': 1.0 if uncertainty_analysis['has_limitations'] else 0.5
        }
        
        credibility_score = sum(weights[key] * components[key] for key in weights.keys())
        return min(credibility_score, 1.0)
    
    def build_enhanced_scoring_prompt(self, signals: Dict[str, Dict], validation_reports: Dict[str, Dict], 
                                    historical_returns: Optional[Dict[str, float]] = None) -> str:
        """
        构建增强的评分提示词，包含数据验证信息
        """
        base_prompt = self.build_scoring_prompt(signals, historical_returns)
        
        # 添加数据质量评估信息
        quality_section = "\n\n=== 数据质量评估报告 ===\n"
        
        for signal_id, validation_result in validation_reports.items():
            quality_section += f"\n信号 {signal_id}:\n"
            quality_section += f"- 总体质量分数: {validation_result['quality_score']:.3f}\n"
            quality_section += f"- 可信度分数: {validation_result['credibility_score']:.3f}\n"
            quality_section += f"- 数据有效性: {'有效' if validation_result['is_valid'] else '无效'}\n"
            
            if validation_result['issues']:
                quality_section += f"- 质量问题: {'; '.join(validation_result['issues'])}\n"
            
            # 详细质量指标
            details = validation_result['validation_details']
            if 'citation_quality' in details:
                cq = details['citation_quality']
                quality_section += f"- 数据引用质量: {cq['citation_ratio']:.2%} ({cq['cited_evidence']}/{cq['total_evidence']})\n"
            
            if 'source_diversity' in details:
                sd = details['source_diversity']
                quality_section += f"- 数据源多样性: {sd['diversity_score']:.3f} (使用了{sd['sources_count']}个数据源)\n"
            
            if 'logical_coherence' in details:
                lc = details['logical_coherence']
                quality_section += f"- 逻辑一致性: {lc['coherence_score']:.3f}\n"
                if lc['issues']:
                    quality_section += f"  逻辑问题: {'; '.join(lc['issues'])}\n"
        
        # 增强评分指导
        enhanced_guidance = "\n\n=== 增强评分指导 ===\n"
        enhanced_guidance += "在评分时，请综合考虑以下因素：\n"
        enhanced_guidance += "1. 信号的盈利潜力和市场机会\n"
        enhanced_guidance += "2. 数据质量和可信度分数\n"
        enhanced_guidance += "3. 证据的充分性和引用规范性\n"
        enhanced_guidance += "4. 逻辑推理的一致性和合理性\n"
        enhanced_guidance += "5. 不确定性的适当处理\n\n"
        enhanced_guidance += "评分调整原则：\n"
        enhanced_guidance += "- 数据质量分数 < 0.5: 最高分数不超过70分\n"
        enhanced_guidance += "- 可信度分数 < 0.6: 最高分数不超过75分\n"
        enhanced_guidance += "- 存在严重逻辑问题: 扣除10-20分\n"
        enhanced_guidance += "- 缺乏数据引用: 扣除5-15分\n"
        
        return base_prompt + quality_section + enhanced_guidance
    
    def _apply_validation_adjustments(self, llm_scores: Dict[str, float], 
                                    validation_reports: Dict[str, Dict]) -> Dict[str, float]:
        """
        基于验证结果调整LLM评分
        """
        adjusted_scores = {}
        
        for signal_id, llm_score in llm_scores.items():
            validation_result = validation_reports.get(signal_id, {})
            
            if not validation_result:
                adjusted_scores[signal_id] = llm_score
                continue
            
            quality_score = validation_result.get('quality_score', 0.5)
            credibility_score = validation_result.get('credibility_score', 0.5)
            is_valid = validation_result.get('is_valid', True)
            
            # 基础调整
            adjusted_score = llm_score
            
            # 1. 数据质量调整
            if quality_score < 0.3:
                adjusted_score *= 0.6  # 严重质量问题，大幅降分
            elif quality_score < 0.5:
                adjusted_score *= 0.8  # 质量问题，适度降分
            elif quality_score > 0.8:
                adjusted_score *= 1.1  # 高质量，适度加分
            
            # 2. 可信度调整
            if credibility_score < 0.4:
                adjusted_score *= 0.7
            elif credibility_score < 0.6:
                adjusted_score *= 0.9
            elif credibility_score > 0.8:
                adjusted_score *= 1.05
            
            # 3. 有效性调整
            if not is_valid:
                adjusted_score *= 0.5  # 无效信号严重降分
            
            # 4. 应用硬性限制
            if quality_score < 0.5:
                adjusted_score = min(adjusted_score, 70.0)
            if credibility_score < 0.6:
                adjusted_score = min(adjusted_score, 75.0)
            
            # 确保分数在合理范围内
            adjusted_score = max(min(adjusted_score, 100.0), 0.0)
            
            adjusted_scores[signal_id] = adjusted_score
            
            # 记录调整信息
            if abs(adjusted_score - llm_score) > 1.0:
                self.logger.info(f"Signal {signal_id} score adjusted: {llm_score:.1f} -> {adjusted_score:.1f} "
                               f"(quality: {quality_score:.3f}, credibility: {credibility_score:.3f})")
        
        return adjusted_scores
    
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
researcher_0: 75|Average historical performance(-15), insufficient analysis depth(-10), moderate evidence(-5)
...
researcher_19: 45|Poor historical performance(-25), confused analysis logic(-15), insufficient evidence(-10), missing risk assessment(-5)
researcher_v2_0: 60|Average historical performance(-20), shallow analysis logic(-10), poor evidence quality(-10)
...
researcher_v2_19: 25|Very poor historical performance(-30), confused analysis logic(-20), severely insufficient evidence(-15), missing risk assessment(-10)

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
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        
        try:
            print(f"调用judger_{judger_id} (GLOBAL_LLM)...")
            
            result = GLOBAL_LLM.run(messages, max_tokens=10000, temperature=0.1)
            
            if result and hasattr(result, 'content'):
                return result.content
            else:
                print(f"警告: judger_{judger_id} 响应格式异常")
                return f"错误: 无法解析响应内容"
                
        except Exception as e:
            print(f"错误: judger_{judger_id} 调用失败: {e}")
            return f"错误: {e}"
    
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
                    background_information = agent.build_background_information(trigger_time, agent.config.belief, factors)
                    
                    # 创建agent输入
                    agent_input = ResearchAgentInput(
                        trigger_time=trigger_time,
                        background_information=background_information
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
            
            # 遍历所有agent目录
            reports_dir = self.workspace_dir / "reports"
            if reports_dir.exists():
                for agent_dir in reports_dir.iterdir():
                    if agent_dir.is_dir() and agent_dir.name.startswith('agent_'):
                        agent_name = agent_dir.name
                        returns = []
                        
                        # 获取过去window_m天的信号
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
                            historical_returns[agent_name] = None
            
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
                print(f"信号数据不完整: action={action}, symbol_code={symbol_code}")
                return None
            
            print(f"计算{symbol_code}的5日收益率，信号时间: {signal_time}, 操作: {action}")
            
            # 获取过去5个交易日的价格数据（需要6个点：T-5到T0）
            open_prices = []
            primary_market = market_manager.config.target_markets[0] if getattr(market_manager, 'config', None) and market_manager.config.target_markets else "CN-Stock"
            for i in range(6):  # 需要6个数据点来计算5个交易日的收益率
                try:
                    price_data = market_manager.get_symbol_price(primary_market, symbol_code, signal_time, -i)
                    if not price_data:
                        print(f"  T-{i}: 无法获取价格数据")
                        break
                    
                    open_price = price_data.get('open')
                    trade_date = price_data.get('trade_date', f'Day-{i}')
                    if open_price is not None and open_price > 0:
                        open_prices.append(open_price)
                        print(f"  T-{i}: {trade_date} 开盘价 {open_price:.2f}")
                    else:
                        print(f"  T-{i}: 开盘价无效 {open_price}")
                        break
                except Exception as e:
                    print(f"  T-{i}: 获取价格异常 {e}")
                    break
            
            # 需要至少6个价格点来计算5个交易日收益率
            if len(open_prices) < 6:
                print(f"数据不足，仅获取到{len(open_prices)}个价格点，需要6个")
                # 如果数据不足，尝试计算可用天数的收益率
                if len(open_prices) >= 2:
                    print(f"使用{len(open_prices)-1}个交易日计算收益率")
                    start_price = open_prices[-1]  # 最早的开盘价
                    end_price = open_prices[0]     # 当前日的开盘价
                    
                    # 计算基础收益率
                    base_return = (end_price - start_price) / start_price
                    print(f"  基础收益率: ({end_price:.2f} - {start_price:.2f}) / {start_price:.2f} = {base_return:.4f}")
                    
                    # 根据action调整收益率
                    if action.lower() == 'buy':
                        # buy信号：股价上涨为正收益
                        final_return = base_return
                        print(f"  买入信号，保持收益率: {final_return:.4f}")
                    elif action.lower() == 'sell':
                        # sell信号：股价下跌为正收益，所以取负值
                        final_return = -base_return
                        print(f"  卖出信号，收益率取反: {final_return:.4f}")
                    else:
                        print(f"  未知操作类型: {action}")
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
            print(f"  完整5日收益率: ({end_price:.2f} - {start_price:.2f}) / {start_price:.2f} = {base_return:.4f}")
            
            # 根据action调整收益率
            if action.lower() == 'buy':
                # buy信号：股价上涨为正收益
                final_return = base_return
                print(f"  买入信号，保持收益率: {final_return:.4f}")
            elif action.lower() == 'sell':
                # sell信号：股价下跌为正收益，所以取负值
                final_return = -base_return
                print(f"  卖出信号，收益率取反: {final_return:.4f}")
            else:
                print(f"  未知操作类型: {action}")
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
            reports_dir = self.workspace_dir / "reports"
            if reports_dir.exists():
                for agent_dir in reports_dir.iterdir():
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
        
        # 执行数据质量验证
        print("执行数据质量验证...")
        validation_reports = {}
        for signal_id, signal_data in converted_signals.items():
            validation_result = self.validate_signal_data_quality(signal_data)
            validation_reports[signal_id] = validation_result
            
            # 输出验证摘要
            print(f"  {signal_id}: 质量={validation_result['quality_score']:.3f}, "
                  f"可信度={validation_result['credibility_score']:.3f}, "
                  f"有效={'是' if validation_result['is_valid'] else '否'}")
            
            if validation_result['issues']:
                print(f"    问题: {'; '.join(validation_result['issues'])}")
        
        # 不再计算当天信号标的的历史表现，改为在权重优化阶段计算agent历史信号执行结果
        historical_returns = None
        
        # 构建增强的评分提示词
        prompt = self.build_enhanced_scoring_prompt(converted_signals, validation_reports, historical_returns)
        
        # 并发调用多个judger
        all_scores = {}
        all_responses = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_judgers) as executor:
            # 提交所有judger任务
            future_to_judger = {}
            for judger_id in range(self.num_judgers):
                future = executor.submit(self._score_with_single_judger, judger_id, prompt, validation_reports)
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
    
    def _score_with_single_judger(self, judger_id: int, prompt: str, validation_reports: Dict[str, Dict]) -> Tuple[str, Dict]:
        """单个judger评分的辅助方法，包含验证调整"""
        response = self.call_llm_for_scoring(prompt, judger_id)
        raw_scores = self.parse_llm_scores(response)
        
        # 提取分数值用于验证调整
        llm_scores = {signal_id: score_data['score'] for signal_id, score_data in raw_scores.items()}
        
        # 应用验证调整
        adjusted_scores = self._apply_validation_adjustments(llm_scores, validation_reports)
        
        # 更新分数数据结构
        final_scores = {}
        for signal_id, score_data in raw_scores.items():
            adjusted_score = adjusted_scores.get(signal_id, score_data['score'])
            final_scores[signal_id] = {
                'score': adjusted_score,
                'reason': score_data.get('reason', '无评分理由'),
                'original_score': score_data['score'],
                'adjustment': adjusted_score - score_data['score']
            }
        
        return response, final_scores
    
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


class JudgerCritic:
    """信号评分和权重优化的主控制器"""
    
    def __init__(self, workspace_dir: str = None):
        if workspace_dir is None:
            workspace_dir = PROJECT_ROOT / "agents_workspace"
        
        self.workspace_dir = Path(workspace_dir)
        self.signal_judger = SignalJudger(str(self.workspace_dir))
        self.weight_optimizer = WeightOptimizer(str(self.workspace_dir))
    
    async def run_judger_critic(self, trigger_time: str, research_agents=None) -> Dict[str, Any]:
        """
        运行完整的评分和权重优化流程
        
        Args:
            trigger_time: 触发时间
            research_agents: research agents实例，用于补全缺失信号
            
        Returns:
            Dict: 包含评分结果和权重的完整结果
        """
        print(f"🤖 开始运行JudgerCritic流程，时间: {trigger_time}")
        
        try:
            # 0. 检查并补全缺失信号
            print("🔍 步骤0: 检查历史信号完整性...")
            missing_dates = self.signal_judger.check_missing_signals(trigger_time, self.signal_judger.window_m)
            
            if missing_dates:
                print(f"发现 {len(missing_dates)} 个缺失信号，开始补全...")
                if research_agents:
                    success = await self.signal_judger.run_missing_signals(missing_dates, research_agents)
                    if not success:
                        print("❌ 缺失信号补全失败")
                        return {
                            'status': 'failed',
                            'reason': '缺失信号补全失败',
                            'trigger_time': trigger_time
                        }
                else:
                    print("⚠️ 未提供research_agents，跳过缺失信号补全")
            else:
                print("✅ 历史信号完整，无需补全")
            
            # 1. 信号评分
            print("📊 步骤1: 信号评分...")
            all_scores, all_responses = await self.signal_judger.judge_signals(trigger_time)
            
            if not all_scores:
                print("⚠️ 没有获得评分结果，退出")
                return {
                    'status': 'failed',
                    'reason': '没有获得评分结果',
                    'trigger_time': trigger_time
                }
            
            # 2. 计算共识评分
            print("🔄 步骤2: 计算共识评分...")
            consensus_scores = self.weight_optimizer.calculate_consensus_scores(all_scores)
            
            # 2.5. 过滤无效信号 (has_opportunity=no)
            print("🔍 步骤2.5: 过滤无效信号...")
            signals = self.signal_judger.data_converter.load_research_signals(trigger_time)
            factors = self.signal_judger.data_converter.load_factor_data(trigger_time)
            converted_signals = self.signal_judger.data_converter.convert_signals_for_judging(signals, factors)
            
            # 过滤掉has_opportunity=no的信号
            valid_signals = {}
            filtered_consensus_scores = {}
            for signal_name, signal_data in converted_signals.items():
                has_opportunity = signal_data.get('has_opportunity', 'no')
                if has_opportunity.lower() == 'yes':
                    valid_signals[signal_name] = signal_data
                    if signal_name in consensus_scores:
                        filtered_consensus_scores[signal_name] = consensus_scores[signal_name]
                    print(f"   ✅ 保留有效信号: {signal_name} (has_opportunity={has_opportunity})")
                else:
                    print(f"   ❌ 过滤无效信号: {signal_name} (has_opportunity={has_opportunity})")
            
            print(f"   过滤前信号数量: {len(consensus_scores)}, 过滤后有效信号数量: {len(filtered_consensus_scores)}")
            consensus_scores = filtered_consensus_scores
            
            # 3. 权重优化（基于共识评分和历史收益率）
            print("⚖️ 步骤3: 权重优化...")
            optimized_weights = self.weight_optimizer.optimize_weights(consensus_scores, trigger_time)
            
            # 4. 保存最终结果
            print("💾 步骤4: 保存最终结果...")
            final_result = self.weight_optimizer.save_final_results(
                trigger_time, consensus_scores, optimized_weights
            )
            
            print("✅ JudgerCritic流程完成")
            print(f"   共识评分数量: {len(consensus_scores)}")
            print(f"   平均评分: {final_result['summary']['avg_score']:.2f}")
            print(f"   最高评分信号: {final_result['summary']['top_signals'][0] if final_result['summary']['top_signals'] else 'None'}")
            
            return {
                'status': 'success',
                'trigger_time': trigger_time,
                'all_scores': all_scores,
                'consensus_scores': consensus_scores,
                'optimized_weights': optimized_weights,
                'final_result': final_result
            }
            
        except Exception as e:
            print(f"❌ JudgerCritic流程失败: {e}")
            return {
                'status': 'failed',
                'reason': str(e),
                'trigger_time': trigger_time
            }


# 主函数用于测试
async def main():
    """测试函数"""
    judger_critic = JudgerCritic()
    
    # 使用示例时间进行测试
    test_time = "2025-08-07 09:00:00"
    result = await judger_critic.run_judger_critic(test_time)
    
    print("\n" + "="*60)
    print("测试结果:")
    print(f"状态: {result['status']}")
    if result['status'] == 'success':
        print(f"共识评分: {result['consensus_scores']}")
        print(f"优化权重: {result['optimized_weights']}")
    else:
        print(f"失败原因: {result['reason']}")


if __name__ == "__main__":
    asyncio.run(main())
