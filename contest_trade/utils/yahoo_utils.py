"""
Yahoo Finance 工具函数 - 免费获取美股数据
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import requests
from bs4 import BeautifulSoup
import time


def get_yahoo_stock_price(symbol: str, start_date: str, end_date: str, verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    获取Yahoo Finance股价数据
    
    Args:
        symbol: 股票代码，如 'AAPL'
        start_date: 开始日期 '2024-01-01'
        end_date: 结束日期 '2024-01-31'
        verbose: 是否打印详细信息
    
    Returns:
        DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume']
    """
    try:
        if verbose:
            print(f"获取 {symbol} 从 {start_date} 到 {end_date} 的股价数据...")
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            if verbose:
                print(f"未找到 {symbol} 的数据")
            return None
        
        # 重置索引，将日期作为列
        df = hist.reset_index()
        
        # 标准化列名
        df.columns = [col.lower() for col in df.columns]
        df = df.rename(columns={
            'adj close': 'adj_close'
        })
        
        # 确保date列为datetime类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif df.index.name == 'Date':
            df = df.reset_index()
            df['date'] = pd.to_datetime(df['Date'])
            df = df.drop('Date', axis=1)
        
        if verbose:
            print(f"成功获取 {len(df)} 条数据")
        
        return df[['date', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        if verbose:
            print(f"获取 {symbol} 数据失败: {e}")
        return None


def get_yahoo_stock_info(symbol: str, verbose: bool = False) -> Optional[dict]:
    """
    获取Yahoo Finance股票基本信息
    
    Args:
        symbol: 股票代码
        verbose: 是否打印详细信息
    
    Returns:
        包含公司信息的字典
    """
    try:
        if verbose:
            print(f"获取 {symbol} 的基本信息...")
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # 提取关键信息
        key_info = {
            'symbol': symbol,
            'company_name': info.get('longName', symbol),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'current_price': info.get('currentPrice', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'description': info.get('businessSummary', '')[:500] + '...' if info.get('businessSummary') else ''
        }
        
        if verbose:
            print(f"成功获取 {key_info['company_name']} 信息")
        
        return key_info
        
    except Exception as e:
        if verbose:
            print(f"获取 {symbol} 信息失败: {e}")
        return None


def get_yahoo_financial_news(symbols: List[str] = None, limit: int = 10, verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    获取Yahoo Finance财经新闻
    
    Args:
        symbols: 股票代码列表，如果为None则获取通用新闻
        limit: 新闻数量限制
        verbose: 是否打印详细信息
    
    Returns:
        DataFrame with columns: ['title', 'summary', 'published', 'url']
    """
    try:
        if verbose:
            print("获取Yahoo Finance财经新闻...")
        
        news_data = []
        
        if symbols:
            # 获取特定股票新闻
            for symbol in symbols[:3]:  # 限制3个股票避免过多请求
                try:
                    ticker = yf.Ticker(symbol)
                    news = ticker.news
                    
                    for item in news[:limit//len(symbols)]:
                        content = item.get('content', {})
                        news_data.append({
                            'title': content.get('title', ''),
                            'summary': content.get('summary', '')[:300] + '...' if content.get('summary') else '',
                            'published': content.get('pubDate', ''),
                            'url': content.get('canonicalUrl', {}).get('url', ''),
                            'source': content.get('provider', {}).get('displayName', 'Yahoo Finance')
                        })
                except:
                    continue
        else:
            # 获取市场通用新闻 - 使用SPY作为代表
            try:
                ticker = yf.Ticker('SPY')
                news = ticker.news
                
                for item in news[:limit]:
                    content = item.get('content', {})
                    news_data.append({
                        'title': content.get('title', ''),
                        'summary': content.get('summary', '')[:300] + '...' if content.get('summary') else '',
                        'published': content.get('pubDate', ''),
                        'url': content.get('canonicalUrl', {}).get('url', ''),
                        'source': content.get('provider', {}).get('displayName', 'Yahoo Finance')
                    })
            except:
                pass
        
        if not news_data:
            if verbose:
                print("未获取到新闻数据")
            return None
        
        df = pd.DataFrame(news_data)
        
        # 转换时间戳为日期时间 - 新的API返回的是ISO格式字符串，不是时间戳
        df['published'] = pd.to_datetime(df['published'], errors='coerce')
        
        if verbose:
            print(f"成功获取 {len(df)} 条新闻")
        
        return df
        
    except Exception as e:
        if verbose:
            print(f"获取新闻失败: {e}")
        return None


def get_yahoo_market_summary(verbose: bool = False) -> Optional[dict]:
    """
    获取Yahoo Finance市场概况
    
    Returns:
        包含主要指数信息的字典
    """
    try:
        if verbose:
            print("获取市场概况...")
        
        # 主要指数
        indices = {
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'Dow Jones',
            '^RUT': 'Russell 2000'
        }
        
        market_data = {}
        
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else latest
                    
                    change = latest['Close'] - prev['Close']
                    change_pct = (change / prev['Close']) * 100
                    
                    market_data[name] = {
                        'price': round(latest['Close'], 2),
                        'change': round(change, 2),
                        'change_pct': round(change_pct, 2),
                        'volume': int(latest['Volume'])
                    }
            except:
                continue
        
        if verbose:
            print(f"成功获取 {len(market_data)} 个指数数据")
        
        return market_data
        
    except Exception as e:
        if verbose:
            print(f"获取市场概况失败: {e}")
        return None


# 测试函数
if __name__ == "__main__":
    # 测试股价获取
    print("测试股价获取...")
    df = get_yahoo_stock_price('AAPL', '2024-08-14', '2024-08-15', verbose=True)
    if df is not None:
        print(df.head())
    
    # 测试公司信息
    print("\n测试公司信息...")
    info = get_yahoo_stock_info('AAPL', verbose=True)
    if info:
        print(f"公司: {info['company_name']}")
        print(f"行业: {info['sector']} - {info['industry']}")
    
    # 测试新闻获取
    print("\n测试新闻获取...")
    news = get_yahoo_financial_news(['AAPL'], limit=3, verbose=True)
    if news is not None:
        print(news[['title', 'source']].head())
    
    # 测试市场概况
    print("\n测试市场概况...")
    market = get_yahoo_market_summary(verbose=True)
    if market:
        for name, data in market.items():
            print(f"{name}: {data['price']} ({data['change_pct']:+.2f}%)")
