"""
数据格式转换器
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

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

