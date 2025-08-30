"""
polygon 的工具函数
1. 获取美股新闻
"""
import sys
from pathlib import Path
import os
import json
import pandas as pd
import requests
from functools import lru_cache
import hashlib
import pickle
import time
from typing import List
from config.config import cfg

DEFAULT_POLYGON_CACHE_DIR = Path(__file__).parent / "polygon_cache"

class CachedPolygonClient:
    def __init__(self, cache_dir=None, api_key=None):
        if not cache_dir:
            self.cache_dir = DEFAULT_POLYGON_CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取API密钥
        if not api_key:
            api_key = cfg.polygon_key
        
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2"
        self.rate_limit_delay = 0.2  # API限制，每秒最多5次请求

    def run(self, endpoint: str, params: dict, verbose: bool = False):
        """
        运行polygon API请求并缓存结果
        
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
        endpoint_clean = endpoint.replace('/', '_').replace('?', '_').replace(':', '_').replace('=', '_').replace('&', '_').replace('-', '_').lstrip('_')
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
                params['apiKey'] = self.api_key
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


    def get_stock_news(self, symbol: str = None, limit: int = 1000, verbose: bool = False):
        """
        获取股票新闻
        
        Args:
            symbol: 股票代码 (可选)
            limit: 返回新闻数量
        """
        endpoint = '/reference/news'
        params = {'order': 'desc', 'limit': limit, 'sort': 'published_utc'}
        if symbol:
            params['ticker'] = symbol
        return self.run(endpoint, params, verbose=verbose)


# 创建全局缓存客户端
polygon_cached = CachedPolygonClient()


@lru_cache(maxsize=1000)
def get_us_stock_news(symbol: str = None, limit: int = 1000, verbose: bool = False):
    """获取美股新闻"""
    result = polygon_cached.get_stock_news(symbol, limit, verbose)
    df = process_polygon_news(result)
    if df is not None and not df.empty and 'published_utc' in df.columns:
        df['published_utc'] = pd.to_datetime(df['published_utc'])
        df = df.sort_values('published_utc', ascending=False).reset_index(drop=True)
        return df
    return pd.DataFrame()


def process_polygon_news(result: dict):
    "处理响应结果中的新闻"
    if not result:
        print("Fail to get polygon news")
        return pd.DataFrame()

    items = result.get('results') if isinstance(result, dict) else result
    if not items:
        return pd.DataFrame()

    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = (item.get('title') or '').replace('\n', ' ').strip()
        published = (item.get('published_utc') or '').strip()
        url = (item.get('article_url') or '').strip()
        desc = (item.get('description') or '').replace('\n', ' ').strip()
        rows.append({
            'title': title,
            'published_utc': published,
            'article_url': url,
            'description': desc,
        })
    df = pd.DataFrame(rows, columns=['title', 'published_utc', 'article_url', 'description'])
    return df

if __name__ == "__main__":
    df = get_us_stock_news()
    print(df)
    records_df = df.copy()
    if 'published_utc' in records_df.columns:
        if pd.api.types.is_datetime64_any_dtype(records_df['published_utc']):
            records_df['published_utc'] = records_df['published_utc'].astype(str)  # 或 .dt.strftime('%Y-%m-%d %H:%M:%S%z')
    with open("polygon_news.json", "w", encoding="utf-8") as f:
        json.dump(records_df.to_dict(orient='records'), f, ensure_ascii=False, indent=4)