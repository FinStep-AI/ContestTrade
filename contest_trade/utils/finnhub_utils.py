"""
finnhub 的工具函数

1. 获取美股财务数据
2. 获取美股价格数据  
3. 获取美股基本信息
"""
import os
import json
import pandas as pd
import finnhub
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import pickle
import time
from config.config import cfg

DEFAULT_FINNHUB_CACHE_DIR = Path(__file__).parent / "finnhub_cache"

class CachedFinnhubClient:
    def __init__(self, cache_dir=None, api_key=None):
        if not cache_dir:
            self.cache_dir = DEFAULT_FINNHUB_CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取API密钥
        if not api_key:
            api_key = cfg.finnhub_key
        
        self.client = finnhub.Client(api_key=api_key)
        self.rate_limit_delay = 1.0  # API限制，每秒最多60次请求

    def run(self, func_name: str, func_kwargs: dict, verbose: bool = False):
        """
        运行finnhub客户端方法并缓存结果
        
        Args:
            func_name: finnhub客户端方法名
            func_kwargs: 方法参数
            verbose: 是否输出详细信息
        """
        func_kwargs_str = json.dumps(func_kwargs, sort_keys=True)
        return self.run_with_cache(func_name, func_kwargs_str, verbose)
    
    def run_with_cache(self, func_name: str, func_kwargs: str, verbose: bool = False):
        func_kwargs = json.loads(func_kwargs)
        
        # 创建缓存文件路径
        args_hash = hashlib.md5(str(func_kwargs).encode()).hexdigest()
        func_cache_dir = self.cache_dir / func_name
        if not func_cache_dir.exists():
            func_cache_dir.mkdir(parents=True, exist_ok=True)
        func_cache_file = func_cache_dir / f"{args_hash}.pkl"
        
        # 尝试从缓存加载
        if func_cache_file.exists():
            if verbose:
                print(f"📁 从缓存加载: {func_cache_file}")
            with open(func_cache_file, "rb") as f:
                return pickle.load(f)
        else:
            if verbose:
                print(f"🌐 API请求: {func_name} 参数: {func_kwargs}")
            
            # 限制API请求频率
            time.sleep(self.rate_limit_delay)
            
            try:
                # 调用finnhub客户端方法
                result = getattr(self.client, func_name)(**func_kwargs)
                
                # 保存到缓存
                if verbose:
                    print(f"💾 保存缓存: {func_cache_file}")
                with open(func_cache_file, "wb") as f:
                    pickle.dump(result, f)
                
                return result
            except Exception as e:
                if verbose:
                    print(f"❌ API请求失败: {e}")
                raise e

    def get_financials(self, symbol: str, statement: str = 'ic', freq: str = 'annual', verbose: bool = False):
        """
        获取财务数据
        
        Args:
            symbol: 股票代码 (如 'AAPL')
            statement: 财务报表类型 ('ic'=损益表, 'bs'=资产负债表, 'cf'=现金流量表)
            freq: 频率 ('annual'=年报, 'quarterly'=季报)
        """
        return self.run('financials', {
            'symbol': symbol,
            'statement': statement,
            'freq': freq
        }, verbose=verbose)

    def get_quote(self, symbol: str, verbose: bool = False):
        """获取实时报价"""
        return self.run('quote', {'symbol': symbol}, verbose=verbose)

    def get_candles(self, symbol: str, resolution: str = 'D', 
                   from_timestamp: int = None, to_timestamp: int = None, verbose: bool = False):
        """
        获取K线数据
        
        Args:
            symbol: 股票代码
            resolution: 时间周期 ('1', '5', '15', '30', '60', 'D', 'W', 'M')
            from_timestamp: 开始时间戳
            to_timestamp: 结束时间戳
        """
        if from_timestamp is None:
            # 默认获取过去一年的数据
            to_timestamp = int(datetime.now().timestamp())
            from_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
        
        return self.run('stock_candles', {
            'symbol': symbol,
            'resolution': resolution,
            '_from': from_timestamp,  # 注意这里使用 _from 而不是 from
            'to': to_timestamp
        }, verbose=verbose)

    def get_company_profile(self, symbol: str, verbose: bool = False):
        """获取公司基本信息"""
        return self.run('company_profile2', {'symbol': symbol}, verbose=verbose)

    def get_company_news(self, symbol: str, from_date: str, to_date: str, verbose: bool = False):
        """
        获取公司新闻
        
        Args:
            symbol: 股票代码
            from_date: 开始日期 (YYYY-MM-DD)
            to_date: 结束日期 (YYYY-MM-DD)
        """
        return self.run('company_news', {
            'symbol': symbol,
            'from': from_date,
            'to': to_date
        }, verbose=verbose)

    def get_earnings(self, symbol: str, verbose: bool = False):
        """获取盈利数据"""
        return self.run('earnings', {'symbol': symbol}, verbose=verbose)

    def get_recommendation_trends(self, symbol: str, verbose: bool = False):
        """获取分析师推荐趋势"""
        return self.run('recommendation_trends', {'symbol': symbol}, verbose=verbose)

# 创建全局缓存客户端
finnhub_cached = CachedFinnhubClient()

@lru_cache(maxsize=1000)
def get_us_stock_financials(symbol: str, statement: str = 'ic', freq: str = 'annual', verbose: bool = False):
    """
    获取美股财务数据的便捷函数
    
    Args:
        symbol: 股票代码
        statement: 'ic'(损益表), 'bs'(资产负债表), 'cf'(现金流量表)
        freq: 'annual'(年报), 'quarterly'(季报)
    """
    return finnhub_cached.get_financials(symbol, statement, freq, verbose)

@lru_cache(maxsize=1000)
def get_us_stock_price(symbol: str, verbose: bool = False):
    """获取美股实时价格"""
    return finnhub_cached.get_quote(symbol, verbose)

@lru_cache(maxsize=1000)
def get_us_stock_profile(symbol: str, verbose: bool = False):
    """获取美股公司信息"""
    return finnhub_cached.get_company_profile(symbol, verbose)

@lru_cache(maxsize=1000)
def get_us_stock_candles(symbol: str, days: int = 365, resolution: str = 'D', verbose: bool = False):
    """
    获取美股K线数据
    
    Args:
        symbol: 股票代码
        days: 获取过去多少天的数据
        resolution: 时间周期
    """
    to_timestamp = int(datetime.now().timestamp())
    from_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
    
    result = finnhub_cached.get_candles(symbol, resolution, from_timestamp, to_timestamp, verbose)
    
    # 转换为DataFrame格式
    if result and 's' in result and result['s'] == 'ok':
        df = pd.DataFrame({
            'timestamp': result['t'],
            'open': result['o'],
            'high': result['h'], 
            'low': result['l'],
            'close': result['c'],
            'volume': result['v']
        })
        # 转换时间戳为日期
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
        return df
    return pd.DataFrame()

def format_financial_data(financial_data):
    """
    格式化财务数据为DataFrame
    """
    if not financial_data or 'financials' not in financial_data:
        return pd.DataFrame()
    
    financials = financial_data['financials']
    if not financials:
        return pd.DataFrame()
    
    # 提取所有年份/季度的数据
    all_data = []
    for period_data in financials:
        period = period_data.get('period', '')
        year = period_data.get('year', '')
        quarter = period_data.get('quarter', '')
        
        row_data = {
            'period': period,
            'year': year,
            'quarter': quarter
        }
        
        # 添加所有财务指标
        for item in period_data.get('report', []):
            concept = item.get('concept', '')
            value = item.get('value', 0)
            row_data[concept] = value
        
        all_data.append(row_data)
    
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    pass