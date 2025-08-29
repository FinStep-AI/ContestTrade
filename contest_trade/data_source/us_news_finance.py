"""美股财经新闻数据源
获取美股相关的财经新闻，包括市场动态、公司公告、分析师观点等
代码来源：佐佑的微信好友
"""
import sys
from pathlib import Path

_CURRENT_FILE = Path(__file__).resolve()
_CONTEST_TRADE_DIR = _CURRENT_FILE.parents[1]
if str(_CONTEST_TRADE_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTEST_TRADE_DIR))
import pandas as pd
import asyncio
import traceback
from datetime import datetime, timedelta
from data_source.data_source_base import DataSourceBase
from models.llm_model import GLOBAL_LLM
from loguru import logger
import requests
from bs4 import BeautifulSoup
import feedparser
import re
from urllib.parse import urljoin

class USNewsFinance(DataSourceBase):
    def __init__(self):
        super().__init__("us_news_finance")
        
        # 美股财经新闻RSS源
        self.rss_feeds = [
            {
                "name": "Yahoo Finance",
                "url": "https://feeds.finance.yahoo.com/rss/2.0/headline",
                "category": "market_news"
            },
            {
                "name": "MarketWatch",
                "url": "https://feeds.marketwatch.com/marketwatch/topstories/",
                "category": "market_news"
            },
            {
                "name": "Reuters Business",
                "url": "https://feeds.reuters.com/reuters/businessNews",
                "category": "business_news"
            },
            {
                "name": "CNBC Markets",
                "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
                "category": "market_news"
            }
        ]
        
    async def get_data(self, trigger_time: str) -> pd.DataFrame:
        try:
            df = self.get_data_cached(trigger_time)
            if df is not None:
                return df
            
            logger.info(f"获取美股财经新闻数据: {trigger_time}")
            
            # 获取新闻数据
            news_data = await self.fetch_news_data(trigger_time)
            print(news_data)
            if not news_data:
                logger.warning("未获取到美股新闻数据")
                return pd.DataFrame()
            
            # 使用LLM分析新闻
            llm_summary = await self.get_llm_summary(news_data, trigger_time)
            
            # 构建返回数据
            data = [{
                "title": f"{trigger_time.split(' ')[0]}:美股财经新闻汇总",
                "content": llm_summary,
                "pub_time": trigger_time,
                "url": None
            }]
            
            df = pd.DataFrame(data)
            self.save_data_cached(trigger_time, df)
            return df
                
        except Exception as e:
            logger.error(f"获取美股新闻数据失败: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    async def fetch_news_data(self, trigger_time: str) -> list:
        """获取新闻数据"""
        all_news = []
        target_date = datetime.strptime(trigger_time.split(' ')[0], '%Y-%m-%d')
        
        for feed_info in self.rss_feeds:
            try:
                logger.info(f"获取 {feed_info['name']} 新闻")
                news_items = await self.fetch_rss_news(feed_info, target_date)
                all_news.extend(news_items)
            except Exception as e:
                logger.error(f"获取 {feed_info['name']} 新闻失败: {e}")
                continue
        
        # 按时间排序，取最新的50条
        all_news.sort(key=lambda x: x.get('pub_time', ''), reverse=True)
        return all_news[:50]
    
    async def fetch_rss_news(self, feed_info: dict, target_date: datetime) -> list:
        """获取RSS新闻"""
        try:
            # 设置请求头
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # 获取RSS内容
            response = requests.get(feed_info['url'], headers=headers, timeout=10)
            response.raise_for_status()
            
            # 解析RSS
            feed = feedparser.parse(response.content)
            
            news_items = []
            cutoff_date = target_date - timedelta(days=2)  # 获取2天内的新闻
            
            for entry in feed.entries[:20]:  # 限制每个源最多20条
                try:
                    # 解析发布时间
                    pub_time = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_time = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_time = datetime(*entry.updated_parsed[:6])
                    
                    # 过滤时间范围
                    if pub_time and pub_time < cutoff_date:
                        continue
                    
                    # 提取内容
                    title = entry.get('title', '').strip()
                    summary = entry.get('summary', '').strip()
                    link = entry.get('link', '')
                    
                    # 清理HTML标签
                    if summary:
                        summary = BeautifulSoup(summary, 'html.parser').get_text().strip()
                    
                    # 过滤美股相关新闻
                    if self._is_us_market_related(title, summary):
                        news_items.append({
                            'title': title,
                            'content': summary,
                            'url': link,
                            'pub_time': pub_time.strftime('%Y-%m-%d %H:%M:%S') if pub_time else '',
                            'source': feed_info['name'],
                            'category': feed_info['category']
                        })
                        
                except Exception as e:
                    logger.error(f"解析新闻条目失败: {e}")
                    continue
            
            logger.info(f"从 {feed_info['name']} 获取到 {len(news_items)} 条美股相关新闻")
            return news_items
            
        except Exception as e:
            logger.error(f"获取RSS新闻失败 {feed_info['name']}: {e}")
            return []
    
    def _is_us_market_related(self, title: str, content: str) -> bool:
        """判断是否为美股相关新闻"""
        text = f"{title} {content}".lower()
        
        # 美股相关关键词
        us_keywords = [
            'nasdaq', 'dow', 'sp 500', 's&p 500', 'nyse', 'wall street',
            'federal reserve', 'fed', 'treasury', 'inflation', 'gdp',
            'earnings', 'stock', 'shares', 'market', 'trading',
            'apple', 'microsoft', 'amazon', 'google', 'tesla', 'nvidia',
            'meta', 'netflix', 'berkshire', 'johnson', 'jpmorgan',
            'us economy', 'american', 'united states', 'dollar', 'usd'
        ]
        
        # 检查是否包含美股关键词
        for keyword in us_keywords:
            if keyword in text:
                return True
        
        # 排除明显的非美股新闻
        exclude_keywords = [
            'china', 'chinese', 'shanghai', 'shenzhen', 'hong kong',
            'europe', 'european', 'london', 'tokyo', 'nikkei'
        ]
        
        for keyword in exclude_keywords:
            if keyword in text and 'us' not in text and 'american' not in text:
                return False
        
        return True
    
    async def get_llm_summary(self, news_data: list, trigger_time: str) -> str:
        """使用LLM生成新闻摘要分析"""
        try:
            if not news_data:
                return "未获取到美股新闻数据"
            
            # 构建新闻文本
            news_text = self._construct_news_text(news_data, trigger_time)
            
            # LLM分析提示
            prompt = f"""请基于以下美股财经新闻，生成一份专业的市场新闻分析报告：

{news_text}

请从以下几个维度进行分析：
1. 市场热点和关键事件
2. 重要公司动态和业绩表现
3. 宏观经济政策影响
4. 市场情绪和投资者关注点
5. 潜在的投资机会和风险

要求：
- 分析要客观专业，突出重要信息
- 按重要性排序，优先分析影响较大的新闻
- 提供具体的事实和数据支撑
- 语言简洁明了，逻辑清晰
- 字数控制在800-1200字"""

            try:
                llm_response = await GLOBAL_LLM.acall(prompt)
                return llm_response.strip()
            except Exception as e:
                logger.error(f"LLM分析失败: {e}")
                return f"LLM分析服务不可用。新闻概要：\n{news_text[:1000]}..."
                
        except Exception as e:
            logger.error(f"生成新闻摘要失败: {e}")
            return f"新闻摘要生成失败: {str(e)}"
    
    def _construct_news_text(self, news_data: list, trigger_time: str) -> str:
        """构建用于LLM分析的新闻文本"""
        lines = [f"=== {trigger_time.split(' ')[0]} 美股财经新闻汇总 ==="]
        
        # 按类别分组
        market_news = []
        business_news = []
        
        for news in news_data:
            if news.get('category') == 'market_news':
                market_news.append(news)
            else:
                business_news.append(news)
        
        # 市场新闻
        if market_news:
            lines.append("\n【市场动态】")
            for i, news in enumerate(market_news[:10], 1):
                lines.append(f"{i}. {news['title']}")
                if news.get('content'):
                    lines.append(f"   {news['content'][:200]}...")
                lines.append(f"   来源: {news['source']} | 时间: {news.get('pub_time', 'N/A')}")
                lines.append("")
        
        # 商业新闻
        if business_news:
            lines.append("\n【商业资讯】")
            for i, news in enumerate(business_news[:10], 1):
                lines.append(f"{i}. {news['title']}")
                if news.get('content'):
                    lines.append(f"   {news['content'][:200]}...")
                lines.append(f"   来源: {news['source']} | 时间: {news.get('pub_time', 'N/A')}")
                lines.append("")
        
        return "\n".join(lines)

if __name__ == "__main__":
    us_news = USNewsFinance()
    df = asyncio.run(us_news.get_data("2025-08-28 09:00:00"))
    print(df.head())
    if len(df) > 0:
        print("美股新闻分析内容:")
        print(df.content.values[0])