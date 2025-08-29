"""
SeekingAlpha 的工具函数
1. 获取美股新闻
"""
import sys
from pathlib import Path

_CURRENT_FILE = Path(__file__).resolve()
_CONTEST_TRADE_DIR = _CURRENT_FILE.parents[1]
if str(_CONTEST_TRADE_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTEST_TRADE_DIR))

import os
import json
import pandas as pd
import requests
import re
import html
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import pickle
import time
from typing import List
from config.config import cfg

DEFAULT_SEEKINGALPHA_CACHE_DIR = Path(__file__).parent / "seekingalpha_cache"

class CachedSeekingAlphaClient:
    def __init__(self, cache_dir=None, api_key=None):
        if not cache_dir:
            self.cache_dir = DEFAULT_SEEKINGALPHA_CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取API密钥
        if not api_key:
            api_key = cfg.seekingalpha_key
        
        self.api_key = api_key
        self.base_url = "https://seeking-alpha.p.rapidapi.com"
        self.rate_limit_delay = 0.2  # API限制，每秒最多5次请求

    def run(self, endpoint: str, params: dict, verbose: bool = False):
        """
        运行SeekingAlpha API请求并缓存结果
        
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
                headers = {
                    "x-rapidapi-key": self.api_key,
                    "x-rapidapi-host": "seeking-alpha.p.rapidapi.com"
                }
                # 发送请求
                response = requests.get(url, headers=headers, params=params)
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


    def get_stock_news(self, symbol: str = None, limit: int = 40, verbose: bool = False):
        """
        获取股票新闻
        
        Args:
            symbol: 股票代码 (可选)
            limit: 返回新闻数量
        """
        params = {'size': limit}
        if symbol:
            endpoint = f'/news/v2/list-by-symbol?id={symbol}'
        else:
            endpoint = '/news/v2/list?category=market-news::all'
        return self.run(endpoint, params, verbose=verbose)


    def get_news_content(self, id: str, verbose: bool = False):
        """
        获取新闻具体内容
        """
        endpoint = f'/news/get-details?id={id}'
        return self.run(endpoint, {}, verbose=verbose)
    

    def add_news_with_content(self, news_list: list, verbose: bool = False):
        """
        为新闻列表中的每条新闻添加内容
        
        Args:
            news_list: 新闻列表
            verbose: 是否输出详细信息
            
        Returns:
            list: 包含内容的新闻列表
        """
        completed_news = []
        for item in news_list:
            try:
                # 获取新闻内容
                content_result = self.get_news_content(item["id"], verbose)
                
                # 提取内容文本
                if isinstance(content_result, dict):
                    content = content_result.get("data", {}).get("attributes", {}).get("content", "")
                    if not content:
                        content = content_result.get("content", "")
                else:
                    content = str(content_result) if content_result else ""
                
                # 清理HTML内容
                cleaned_content = clean_html_content(content)
                
                # 创建新的新闻项
                completed_item = item.copy()
                completed_item["content"] = cleaned_content
                completed_news.append(completed_item)
                
                if verbose:
                    print(f"✅ 已获取新闻内容: {item.get('title', 'Unknown')[:50]}...")
                    
            except Exception as e:
                if verbose:
                    print(f"❌ 获取新闻内容失败 (ID: {item.get('id', 'Unknown')}): {e}")
                # 即使获取内容失败，也保留原始新闻项
                completed_item = item.copy()
                completed_item["content"] = ""
                completed_news.append(completed_item)
        
        return completed_news

# 创建全局缓存客户端
seekingalpha_cached = CachedSeekingAlphaClient()

def process_raw_result(raw_result: dict):
    base = "https://seekingalpha.com"
    processed = []
    for item in raw_result.get("data", []):
        attributes = item.get("attributes", {})
        links = item.get("links", {})

        publish_on = attributes.get("publishOn")
        pub_time = None
        if publish_on:
            try:
                dt = datetime.fromisoformat(publish_on)
                pub_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pub_time = publish_on

        processed.append({
            "id": item.get("id", ""),
            "title": attributes.get("title", ""),
            "publishedDate": pub_time,
            "url": (base + links.get("self")) if links.get("self") else None
        })
    return processed


def clean_html_content(content: str):
    text_no_html = re.sub(r"<[^>]+>", "", content)
    clean_content = html.unescape(text_no_html)
    clean_content = re.sub(r"\s+", " ", clean_content).strip()
    return clean_content

@lru_cache(maxsize=1000)
def get_us_stock_news(symbol: str = None, limit: int = 40, verbose: bool = False):
    """获取美股新闻"""
    # 获取新闻列表
    result = seekingalpha_cached.get_stock_news(symbol, limit, verbose)
    result = process_raw_result(result)
    
    # 为每条新闻添加内容
    if result:
        result = seekingalpha_cached.add_news_with_content(result, verbose)
        # print(result)
        # with open("seekingalpha_news.json", "w") as f:
        #     json.dump(result, f, indent=4)
    if result:
        df = pd.DataFrame(result)
        if not df.empty and 'publishedDate' in df.columns:
            df['publishedDate'] = pd.to_datetime(df['publishedDate'])
            df = df.sort_values('publishedDate', ascending=False).reset_index(drop=True)
            return df
    return pd.DataFrame()


if __name__ == "__main__":
    df = get_us_stock_news()
    print(df)