"""
US Polygon News data source
获取美股财经新闻（替代sina_news）
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
from utils.polygon_utils import get_us_stock_news
from loguru import logger


class USPolygonNews(DataSourceBase):
    def __init__(self):
        super().__init__("us_polygon_news")

    async def get_data(self, trigger_time: str) -> pd.DataFrame:
        try:
            # 检查缓存
            cached = self.get_data_cached(trigger_time)
            if cached is not None:
                return cached

            # 获取前一个交易日
            #previous_trading_date = get_previous_trading_date(trigger_time, output_format="%Y-%m-%d")
            previous_trading_date = "20250821"
            # 从polygon获取美股新闻
            news_df = get_us_stock_news(limit=500, verbose=False)
            
            if news_df is None or news_df.empty:
                logger.warning("未获取到polygon美股新闻数据")
                return pd.DataFrame()
            
            # 标准化数据格式
            result_data = []
            for _, row in news_df.iterrows():
                # 处理发布时间
                pub_time = row.get('published_utc', trigger_time)
                if isinstance(pub_time, str):
                    try:
                        # 尝试解析ISO格式时间
                        pub_time = datetime.fromisoformat(pub_time.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pub_time = trigger_time
                
                # 处理内容
                title = str(row.get('title', ''))[:200]  # 限制标题长度
                content = str(row.get('description', ''))[:2000]  # 限制内容长度
                url = str(row.get('article_url', ''))
                
                result_data.append({
                    'title': title,
                    'content': content,
                    'pub_time': pub_time,
                    'url': url
                })
            
            df = pd.DataFrame(result_data)
            print(df)
            self.save_data_cached(trigger_time, df)
            
            logger.info(f"获取美股财经新闻从 {previous_trading_date} 到 {trigger_time} 成功。总计 {len(df)} 条")
            return df
            
        except Exception as e:
            logger.error(f"获取美股财经新闻失败: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    us_news = USPolygonNews()
    print(asyncio.run(us_news.get_data("2025-08-15 10:00:00")))
