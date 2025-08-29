"""
Yahoo Market data source - 免费美股市场数据源
替代us_market.py，使用Yahoo Finance免费API
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
from utils.yahoo_utils import get_yahoo_market_summary, get_yahoo_financial_news
from loguru import logger


class YahooMarket(DataSourceBase):
    def __init__(self):
        super().__init__("yahoo_market")

    def _build_overview(self, trade_date_yyyymmdd: str) -> str:
        """生成基于Yahoo Finance的市场概览"""
        try:
            # 获取市场概况
            market_summary = get_yahoo_market_summary(verbose=False)
            
            # 获取新闻
            news_df = get_yahoo_financial_news(limit=5, verbose=False)
            
            lines = []
            lines.append(f"美股市场概览（{trade_date_yyyymmdd[:4]}-{trade_date_yyyymmdd[4:6]}-{trade_date_yyyymmdd[6:]}）")
            lines.append("数据来源: Yahoo Finance (免费)")
            
            # 主要指数表现
            lines.append("\n📊 主要指数表现：")
            if market_summary:
                for name, data in market_summary.items():
                    change_symbol = "+" if data['change_pct'] >= 0 else ""
                    lines.append(f"- {name}: {data['price']:.2f} ({change_symbol}{data['change_pct']:.2f}%)")
            else:
                lines.append("- 指数数据暂不可用")
            
            # 市场要闻
            lines.append("\n📰 市场要闻：")
            if news_df is not None and not news_df.empty:
                for _, row in news_df.head(3).iterrows():
                    title = str(row['title'] if 'title' in row and pd.notna(row['title']) else '')[:100]
                    source = str(row['source'] if 'source' in row and pd.notna(row['source']) else 'Yahoo Finance')
                    lines.append(f"- {title} ({source})")
            else:
                lines.append("- 暂无重要新闻")
            
            lines.append("\n💡 投资提示：")
            lines.append("- 数据来源于Yahoo Finance免费接口")
            lines.append("- 建议结合基本面分析和技术指标进行判断")
            lines.append("- 投资有风险，决策需谨慎")
            
            return "\n".join(lines).strip()
            
        except Exception as e:
            logger.error(f"生成市场概览失败: {e}")
            return f"市场概览生成失败: {str(e)}"

    async def get_data(self, trigger_time: str) -> pd.DataFrame:
        try:
            # 检查缓存
            cached = self.get_data_cached(trigger_time)
            if cached is not None:
                return cached

            # 获取前一个交易日
            trade_date = get_previous_trading_date(trigger_time)  # YYYYMMDD
            content = self._build_overview(trade_date)
            
            data = [{
                'title': f"{trade_date}: Yahoo Finance Market Overview",
                'content': content,
                'pub_time': trigger_time,
                'url': 'https://finance.yahoo.com',
            }]
            
            df = pd.DataFrame(data)
            
            # 缓存结果
            self.save_data_cached(trigger_time, df)
            return df
            
        except Exception as e:
            logger.error(f"获取Yahoo Finance数据失败: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    ds = YahooMarket()
    test_time = "2024-08-15 09:00:00"
    print(asyncio.run(ds.get_data(test_time)))
