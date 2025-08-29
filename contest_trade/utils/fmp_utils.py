"""
FMP (Financial Modeling Prep) 的工具函数

1. 获取美股历史价格数据
2. 获取美股财务报表数据
3. 获取美股公司基本信息
"""
import os
import json
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import pickle
import time
from typing import List
from config.config import cfg

DEFAULT_FMP_CACHE_DIR = Path(__file__).parent / "fmp_cache"

class CachedFMPClient:
    def __init__(self, cache_dir=None, api_key=None):
        if not cache_dir:
            self.cache_dir = DEFAULT_FMP_CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取API密钥
        if not api_key:
            api_key = cfg.fmp_key
        
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.rate_limit_delay = 0.2  # API限制，每秒最多5次请求

    def run(self, endpoint: str, params: dict, verbose: bool = False):
        """
        运行FMP API请求并缓存结果
        
        Args:
            endpoint: API端点路径
            params: 请求参数
            verbose: 是否输出详细信息
        """
        params_str = json.dumps(params, sort_keys=True)
        return self.run_with_cache(endpoint, params_str, verbose)
    
    def run_with_cache(self, endpoint: str, params_str: str, verbose: bool = False):
        params = json.loads(params_str)
        
        # 创建缓存文件路径
        endpoint_clean = endpoint.replace('/', '_').lstrip('_')  # 清理endpoint路径
        cache_key = f"{endpoint_clean}_{hashlib.md5(params_str.encode()).hexdigest()}"
        endpoint_cache_dir = self.cache_dir / endpoint_clean
        if not endpoint_cache_dir.exists():
            endpoint_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = endpoint_cache_dir / f"{cache_key}.pkl"
        
        # 尝试从缓存加载
        if cache_file.exists():
            if verbose:
                print(f"📁 从缓存加载: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        else:
            if verbose:
                print(f"🌐 API请求: {endpoint} 参数: {params}")
            
            # 限制API请求频率
            time.sleep(self.rate_limit_delay)
            
            try:
                # 构建完整URL
                url = f"{self.base_url}{endpoint}"
                params['apikey'] = self.api_key
                
                # 发送请求
                response = requests.get(url, params=params)
                response.raise_for_status()
                result = response.json()
                
                # 保存到缓存
                if verbose:
                    print(f"💾 保存缓存: {cache_file}")
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
                
                return result
            except Exception as e:
                if verbose:
                    print(f"❌ API请求失败: {e}")
                raise e

    def get_historical_price(self, symbol: str, from_date: str = None, to_date: str = None, 
                           adjusted: bool = True, adj_base_date: str = None, verbose: bool = False):
        """
        获取历史价格数据
        
        Args:
            symbol: 股票代码 (如 'AAPL')
            from_date: 开始日期 (YYYY-MM-DD)
            to_date: 结束日期 (YYYY-MM-DD)
            adjusted: 是否返回前复权价格 (True: 前复权, False: 原始价格)
            adj_base_date: 前复权基准日期 (YYYY-MM-DD)，如果为None则使用FMP默认基准
        
        Returns:
            pd.DataFrame: 历史价格数据
        """
        params = {'symbol': symbol}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
            
        result = self.run('/historical-price-full/' + symbol, params, verbose=verbose)
        
        # 转换为DataFrame
        if result and 'historical' in result:
            df = pd.DataFrame(result['historical'])
            if not df.empty:
                # 确保日期列是日期类型并排序
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
                # 如果需要前复权价格，按指定基准日期调整
                if adjusted and 'adjClose' in df.columns:
                    df = self._use_adjusted_prices(df, adj_base_date)
                
            return df
        return pd.DataFrame()
    
    def _use_adjusted_prices(self, df, adj_base_date=None):
        """
        使用前复权价格替换原始价格
        
        Args:
            df: 包含原始价格和adjClose的DataFrame
            adj_base_date: 前复权基准日期 (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: 包含前复权价格的DataFrame
        """
        if df.empty or 'adjClose' not in df.columns:
            return df
        
        df = df.copy()
        
        # 计算原始复权比例（FMP的adjClose相对于close）
        df['fmp_adj_ratio'] = df['adjClose'] / df['close']
        
        if adj_base_date:
            # 如果指定了基准日期，重新计算前复权价格
            try:
                # 找到基准日期的复权因子
                base_row = df[df['date'].dt.strftime('%Y-%m-%d') == adj_base_date]
                if base_row.empty:
                    print(f"警告：基准日期 {adj_base_date} 不在数据范围内，使用FMP默认复权")
                    base_adj_ratio = 1.0
                else:
                    base_adj_ratio = base_row.iloc[0]['fmp_adj_ratio']
                
                # 重新计算以指定日期为基准的前复权价格
                # 公式：new_adj_price = original_price * (fmp_adj_ratio / base_adj_ratio)
                df['custom_adj_ratio'] = df['fmp_adj_ratio'] / base_adj_ratio
                
                # 按新的复权比例调整所有价格
                df['open'] = df['open'] * df['custom_adj_ratio']
                df['high'] = df['high'] * df['custom_adj_ratio']
                df['low'] = df['low'] * df['custom_adj_ratio']
                df['close'] = df['close'] * df['custom_adj_ratio']
                
                # 清理临时列
                df = df.drop(['fmp_adj_ratio', 'custom_adj_ratio'], axis=1)
                
            except Exception as e:
                print(f"自定义复权计算失败，使用FMP默认复权: {e}")
                # 回退到FMP默认复权
                df['open'] = df['open'] * df['fmp_adj_ratio']
                df['high'] = df['high'] * df['fmp_adj_ratio']
                df['low'] = df['low'] * df['fmp_adj_ratio']
                df['close'] = df['adjClose']
                df = df.drop(['fmp_adj_ratio'], axis=1)
        else:
            # 使用FMP默认的复权价格
            df['open'] = df['open'] * df['fmp_adj_ratio']
            df['high'] = df['high'] * df['fmp_adj_ratio']
            df['low'] = df['low'] * df['fmp_adj_ratio']
            df['close'] = df['adjClose']
            df = df.drop(['fmp_adj_ratio'], axis=1)
        
        return df

    def get_quote(self, symbol: str, verbose: bool = False):
        """获取实时报价"""
        return self.run('/quote/' + symbol, {}, verbose=verbose)

    def get_company_profile(self, symbol: str, verbose: bool = False):
        """获取公司基本信息"""
        return self.run('/profile/' + symbol, {}, verbose=verbose)

    def get_income_statement(self, symbol: str, period: str = 'annual', verbose: bool = False):
        """
        获取损益表
        
        Args:
            symbol: 股票代码
            period: 'annual' 或 'quarter'
        """
        return self.run('/income-statement/' + symbol, {'period': period}, verbose=verbose)

    def get_balance_sheet(self, symbol: str, period: str = 'annual', verbose: bool = False):
        """
        获取资产负债表
        
        Args:
            symbol: 股票代码
            period: 'annual' 或 'quarter'
        """
        return self.run('/balance-sheet-statement/' + symbol, {'period': period}, verbose=verbose)

    def get_cash_flow(self, symbol: str, period: str = 'annual', verbose: bool = False):
        """
        获取现金流量表
        
        Args:
            symbol: 股票代码
            period: 'annual' 或 'quarter'
        """
        return self.run('/cash-flow-statement/' + symbol, {'period': period}, verbose=verbose)

    def get_key_metrics(self, symbol: str, period: str = 'annual', verbose: bool = False):
        """
        获取关键财务指标
        
        Args:
            symbol: 股票代码
            period: 'annual' 或 'quarter'
        """
        return self.run('/key-metrics/' + symbol, {'period': period}, verbose=verbose)

    def get_financial_ratios(self, symbol: str, period: str = 'annual', verbose: bool = False):
        """
        获取财务比率
        
        Args:
            symbol: 股票代码
            period: 'annual' 或 'quarter'
        """
        return self.run('/ratios/' + symbol, {'period': period}, verbose=verbose)

    def get_stock_news(self, symbol: str = None, limit: int = 50, verbose: bool = False):
        """
        获取股票新闻
        
        Args:
            symbol: 股票代码 (可选)
            limit: 返回新闻数量
        """
        params = {'limit': limit}
        if symbol:
            endpoint = f'/stock_news?tickers={symbol}'
        else:
            endpoint = '/stock_news'
        return self.run(endpoint, params, verbose=verbose)

    def get_market_cap(self, symbol: str, verbose: bool = False):
        """获取市值信息"""
        return self.run('/market-capitalization/' + symbol, {}, verbose=verbose)

    def get_analyst_estimates(self, symbol: str, verbose: bool = False):
        """获取分析师预估"""
        return self.run('/analyst-estimates/' + symbol, {}, verbose=verbose)

# 创建全局缓存客户端
fmp_cached = CachedFMPClient()

def _convert_date_format(date_str: str) -> str:
    """
    将日期格式从 YYYYMMDD 转换为 YYYY-MM-DD
    
    Args:
        date_str: 日期字符串，支持 'YYYYMMDD' 或 'YYYY-MM-DD' 格式
    
    Returns:
        str: YYYY-MM-DD 格式的日期字符串
    """
    if not date_str:
        return None
        
    # 如果已经是 YYYY-MM-DD 格式，直接返回
    if '-' in date_str and len(date_str) == 10:
        return date_str
    
    # 如果是 YYYYMMDD 格式，转换为 YYYY-MM-DD
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    # 其他格式尝试解析
    try:
        from datetime import datetime
        # 尝试解析不同格式
        for fmt in ['%Y%m%d', '%Y-%m-%d', '%Y/%m/%d']:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        raise ValueError(f"无法解析日期格式: {date_str}")
    except Exception:
        raise ValueError(f"无效的日期格式: {date_str}")

def get_us_stock_price(symbol: str, from_date: str = None, to_date: str = None, 
                      adjusted: bool = True, adj_base_date: str = None, verbose: bool = False):
    """
    获取美股历史价格数据的便捷函数
    
    Args:
        symbol: 股票代码
        from_date: 开始日期，支持 'YYYYMMDD' 或 'YYYY-MM-DD' 格式
        to_date: 结束日期，支持 'YYYYMMDD' 或 'YYYY-MM-DD' 格式  
        adjusted: 是否返回前复权价格 (True: 前复权用于回测, False: 原始价格)
        adj_base_date: 前复权基准日期，支持 'YYYYMMDD' 或 'YYYY-MM-DD' 格式，设为None使用FMP默认复权
    
    Returns:
        pd.DataFrame: 包含 date, open, high, low, close, volume 等列的价格数据
        注意：如果指定adj_base_date，返回以该日期为基准的前复权价格
    """
    # 转换日期格式
    converted_from_date = _convert_date_format(from_date)
    converted_to_date = _convert_date_format(to_date)
    converted_adj_base_date = _convert_date_format(adj_base_date)
    
    return fmp_cached.get_historical_price(symbol, converted_from_date, converted_to_date, 
                                         adjusted, converted_adj_base_date, verbose)

@lru_cache(maxsize=1000)
def get_us_stock_quote(symbol: str, verbose: bool = False):
    """获取美股实时报价"""
    result = fmp_cached.get_quote(symbol, verbose)
    if result and len(result) > 0:
        return result[0]
    return {}

@lru_cache(maxsize=1000)
def get_us_stock_profile(symbol: str, verbose: bool = False):
    """获取美股公司信息"""
    result = fmp_cached.get_company_profile(symbol, verbose)
    if result and len(result) > 0:
        return result[0]
    return {}

@lru_cache(maxsize=1000)
def get_us_stock_financials(symbol: str, statement_type: str = 'income', period: str = 'annual', verbose: bool = False):
    """
    获取美股财务数据的便捷函数
    
    Args:
        symbol: 股票代码
        statement_type: 'income'(损益表), 'balance'(资产负债表), 'cash'(现金流量表)
        period: 'annual'(年报), 'quarter'(季报)
    """
    if statement_type == 'income':
        result = fmp_cached.get_income_statement(symbol, period, verbose)
    elif statement_type == 'balance':
        result = fmp_cached.get_balance_sheet(symbol, period, verbose)
    elif statement_type == 'cash':
        result = fmp_cached.get_cash_flow(symbol, period, verbose)
    else:
        raise ValueError(f"不支持的财务报表类型: {statement_type}")
    
    # 转换为DataFrame格式
    if result:
        df = pd.DataFrame(result)
        if not df.empty:
            # 按日期排序
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date', ascending=False).reset_index(drop=True)
            return df
    return pd.DataFrame()

@lru_cache(maxsize=1000)
def get_us_stock_metrics(symbol: str, period: str = 'annual', verbose: bool = False):
    """获取美股关键财务指标"""
    result = fmp_cached.get_key_metrics(symbol, period, verbose)
    if result:
        df = pd.DataFrame(result)
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False).reset_index(drop=True)
            return df
    return pd.DataFrame()

@lru_cache(maxsize=1000)
def get_us_stock_ratios(symbol: str, period: str = 'annual', verbose: bool = False):
    """获取美股财务比率"""
    result = fmp_cached.get_financial_ratios(symbol, period, verbose)
    if result:
        df = pd.DataFrame(result)
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False).reset_index(drop=True)
            return df
    return pd.DataFrame()

@lru_cache(maxsize=1000)
def get_us_stock_news(symbol: str = None, limit: int = 50, verbose: bool = False):
    """获取美股新闻"""
    result = fmp_cached.get_stock_news(symbol, limit, verbose)
    if result:
        df = pd.DataFrame(result)
        if not df.empty and 'publishedDate' in df.columns:
            df['publishedDate'] = pd.to_datetime(df['publishedDate'])
            df = df.sort_values('publishedDate', ascending=False).reset_index(drop=True)
            return df
    return pd.DataFrame()

def format_price_data(price_data):
    """
    格式化价格数据为标准OHLCV格式
    """
    if price_data.empty:
        return pd.DataFrame()
    
    # 重命名列以符合标准格式
    column_mapping = {
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'date': 'date'
    }
    
    result_df = price_data.rename(columns=column_mapping)
    
    # 确保数值列为float类型
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
    
    return result_df

if __name__ == "__main__":
    pass