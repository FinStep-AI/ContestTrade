"""
Bloomberg News data crawler
获取Bloomberg新闻
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
from utils.fmp_utils import get_us_stock_news
from loguru import logger

class BloombergNewsCrawl(DataSourceBase):
    def __init__(self):
        super().__init__("bloomberg_news_crawl")


    async def get_data(self, trigger_time: str) -> pd.DataFrame:
        #TODO: 获取Bloomberg新闻
        pass