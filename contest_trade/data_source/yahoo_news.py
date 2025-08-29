"""
Yahoo News data source - 免费美股新闻数据源
替代us_financial_news.py，使用Yahoo Finance免费API
返回DataFrame列: ['title', 'content', 'pub_time', 'url']
"""
import sys
from pathlib import Path

_CURRENT_FILE = Path(__file__).resolve()
_CONTEST_TRADE_DIR = _CURRENT_FILE.parents[1]
if str(_CONTEST_TRADE_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTEST_TRADE_DIR))
    
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from data_source.data_source_base import DataSourceBase
from utils.date_utils import get_previous_trading_date
from utils.yahoo_utils import get_yahoo_financial_news
from loguru import logger


class YahooNews(DataSourceBase):
    def __init__(self):
        super().__init__("yahoo_news")

    async def get_data(self, trigger_time: str) -> pd.DataFrame:
        try:
            # 检查缓存
            cached = self.get_data_cached(trigger_time)
            if cached is not None:
                return cached

            # 获取前一个交易日
            previous_trading_date = get_previous_trading_date(trigger_time, output_format="%Y-%m-%d")
            
            # 从Yahoo Finance获取美股新闻
            news_df = get_yahoo_financial_news(limit=20, verbose=False)
            
            if news_df is None or news_df.empty:
                logger.warning("未获取到Yahoo Finance新闻数据")
                return pd.DataFrame()
            
            # 标准化数据格式
            result_data = []
            for _, row in news_df.iterrows():
                # 处理发布时间
                pub_time = row['published'] if 'published' in row and pd.notna(row['published']) else trigger_time
                if isinstance(pub_time, pd.Timestamp):
                    pub_time = pub_time.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(pub_time, str):
                    pub_time = pub_time
                else:
                    pub_time = trigger_time
                
                # 处理内容
                title = str(row['title'] if 'title' in row and pd.notna(row['title']) else '')[:200]  # 限制标题长度
                content = str(row['summary'] if 'summary' in row and pd.notna(row['summary']) else '')[:1500]  # 使用摘要作为内容
                url = str(row['url'] if 'url' in row and pd.notna(row['url']) else '')
                
                result_data.append({
                    'title': title,
                    'content': content,
                    'pub_time': pub_time,
                    'url': url
                })
            
            df = pd.DataFrame(result_data)
            
            # 缓存结果
            self.save_data_cached(trigger_time, df)
            
            logger.info(f"获取Yahoo Finance新闻从 {previous_trading_date} 到 {trigger_time} 成功。总计 {len(df)} 条")
            return df
            
        except Exception as e:
            logger.error(f"获取Yahoo Finance新闻失败: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    news = YahooNews()
    print(asyncio.run(news.get_data("2024-08-15 10:00:00")))
