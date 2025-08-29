import pandas as pd
from config.config import PROJECT_ROOT
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

class DataSourceBase:
    
    def __init__(self, name: str):
        self.name = name
        self.data_cache_dir = Path(PROJECT_ROOT) / "data_source" / "data_cache" / self.name
        if not self.data_cache_dir.exists():
            self.data_cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"DataSource.{name}")
        
        # 数据质量配置
        self.quality_thresholds = {
            'min_rows': 1,  # 最少行数
            'required_columns': ['title', 'content', 'pub_time', 'url'],  # 必需列
            'max_age_hours': 24,  # 数据最大时效（小时）
            'min_content_length': 10,  # 内容最小长度
            'max_null_ratio': 0.3  # 最大空值比例
        }

    def get_data_cached(self, trigger_time: str) -> pd.DataFrame:
        """
        get data from data source, return format should be a pandas dataframe
        including cols: ['title', 'content', 'pub_time', 'url']
        """
        cache_file_name = trigger_time.replace(" ", "_").replace(":", "-")
        cache_file = self.data_cache_dir / f"{cache_file_name}.pkl"
        if cache_file.exists():
            return pd.read_pickle(cache_file)
        else:
            return None

    def save_data_cached(self, trigger_time: str, data: pd.DataFrame): 
        cache_file_name = trigger_time.replace(" ", "_").replace(":", "-")
        cache_file = self.data_cache_dir / f"{cache_file_name}.pkl"
        data.to_pickle(cache_file)

    def validate_data_quality(self, data: pd.DataFrame, trigger_time: str) -> Tuple[bool, Dict[str, any], str]:
        """
        验证数据质量
        
        Returns:
            Tuple[bool, Dict, str]: (是否通过验证, 质量指标, 验证消息)
        """
        if data is None or data.empty:
            return False, {'row_count': 0}, "数据为空"
        
        quality_metrics = {
            'row_count': len(data),
            'column_completeness': {},
            'content_quality': {},
            'data_freshness': {},
            'overall_score': 0.0
        }
        
        validation_messages = []
        score_components = []
        
        # 1. 检查行数
        if len(data) < self.quality_thresholds['min_rows']:
            validation_messages.append(f"数据行数不足: {len(data)} < {self.quality_thresholds['min_rows']}")
            score_components.append(0.0)
        else:
            score_components.append(1.0)
        
        # 2. 检查必需列
        missing_columns = [col for col in self.quality_thresholds['required_columns'] if col not in data.columns]
        if missing_columns:
            validation_messages.append(f"缺少必需列: {missing_columns}")
            score_components.append(0.0)
        else:
            score_components.append(1.0)
            
            # 3. 检查列完整性
            for col in self.quality_thresholds['required_columns']:
                if col in data.columns:
                    null_ratio = data[col].isnull().sum() / len(data)
                    quality_metrics['column_completeness'][col] = 1 - null_ratio
                    
                    if null_ratio > self.quality_thresholds['max_null_ratio']:
                        validation_messages.append(f"列 {col} 空值比例过高: {null_ratio:.2%}")
        
        # 4. 检查内容质量
        if 'content' in data.columns:
            content_lengths = data['content'].fillna('').str.len()
            avg_content_length = content_lengths.mean()
            short_content_ratio = (content_lengths < self.quality_thresholds['min_content_length']).sum() / len(data)
            
            quality_metrics['content_quality'] = {
                'avg_length': avg_content_length,
                'short_content_ratio': short_content_ratio
            }
            
            if short_content_ratio > 0.5:
                validation_messages.append(f"内容过短比例过高: {short_content_ratio:.2%}")
                score_components.append(0.5)
            else:
                score_components.append(1.0)
        
        # 5. 检查数据时效性
        if 'pub_time' in data.columns:
            try:
                trigger_dt = datetime.strptime(trigger_time, "%Y-%m-%d %H:%M:%S")
                pub_times = pd.to_datetime(data['pub_time'], errors='coerce')
                valid_times = pub_times.dropna()
                
                if len(valid_times) > 0:
                    time_diffs = (trigger_dt - valid_times).dt.total_seconds() / 3600  # 转换为小时
                    fresh_data_ratio = (time_diffs <= self.quality_thresholds['max_age_hours']).sum() / len(valid_times)
                    avg_age_hours = time_diffs.mean()
                    
                    quality_metrics['data_freshness'] = {
                        'fresh_data_ratio': fresh_data_ratio,
                        'avg_age_hours': avg_age_hours
                    }
                    
                    if fresh_data_ratio < 0.3:
                        validation_messages.append(f"新鲜数据比例过低: {fresh_data_ratio:.2%}")
                        score_components.append(0.3)
                    else:
                        score_components.append(fresh_data_ratio)
                else:
                    validation_messages.append("无有效的发布时间数据")
                    score_components.append(0.0)
            except Exception as e:
                validation_messages.append(f"时效性检查失败: {str(e)}")
                score_components.append(0.0)
        
        # 计算总体质量分数
        quality_metrics['overall_score'] = sum(score_components) / len(score_components) if score_components else 0.0
        
        # 判断是否通过验证
        is_valid = quality_metrics['overall_score'] >= 0.6 and len(validation_messages) == 0
        
        validation_message = "; ".join(validation_messages) if validation_messages else "数据质量验证通过"
        
        return is_valid, quality_metrics, validation_message
    
    def get_data_with_validation(self, trigger_time: str) -> Tuple[Optional[pd.DataFrame], Dict[str, any]]:
        """
        获取数据并进行质量验证
        
        Returns:
            Tuple[Optional[pd.DataFrame], Dict]: (验证后的数据, 质量报告)
        """
        try:
            # 获取原始数据
            raw_data = self.get_data(trigger_time)
            
            # 验证数据质量
            is_valid, quality_metrics, validation_message = self.validate_data_quality(raw_data, trigger_time)
            
            quality_report = {
                'source_name': self.name,
                'trigger_time': trigger_time,
                'is_valid': is_valid,
                'validation_message': validation_message,
                'quality_metrics': quality_metrics,
                'data_shape': raw_data.shape if raw_data is not None else (0, 0)
            }
            
            self.logger.info(f"数据质量验证完成: {validation_message} (分数: {quality_metrics.get('overall_score', 0):.2f})")
            
            return raw_data if is_valid else None, quality_report
            
        except Exception as e:
            error_msg = f"数据获取失败: {str(e)}"
            self.logger.error(error_msg)
            
            quality_report = {
                'source_name': self.name,
                'trigger_time': trigger_time,
                'is_valid': False,
                'validation_message': error_msg,
                'quality_metrics': {'overall_score': 0.0},
                'data_shape': (0, 0)
            }
            
            return None, quality_report
    
    def get_data(self, trigger_time: str) -> pd.DataFrame:
        """
        get data from data source, return format should be a pandas dataframe
        including cols: ['title', 'content', 'pub_time', 'url']
        """
        pass

if __name__ == "__main__":
    pass