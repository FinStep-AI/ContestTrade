"""
US Market Insight data source
获取美股市场深度分析（替代thx_news）
基于FMP的市场数据和新闻，生成市场洞察报告
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
from utils.yahoo_utils import get_yahoo_stock_price, get_yahoo_financial_news
from loguru import logger


class USMarketInsight(DataSourceBase):
    def __init__(self):
        super().__init__("us_market_insight")

    def _generate_market_insight(self, trade_date: str, trigger_time: str) -> str:
        """生成市场洞察内容"""
        try:
            # 主要指数
            indices = ['SPY', 'QQQ', 'DIA', 'IWM']  # S&P500, NASDAQ, DOW, Russell2000
            
            # 获取指数表现
            index_performance = []
            for symbol in indices:
                try:
                    # 获取近5日数据
                    end_date = datetime.strptime(trade_date, "%Y%m%d")
                    start_date = end_date - timedelta(days=7)
                    
                    df = get_yahoo_stock_price(symbol, start_date.strftime("%Y-%m-%d"), 
                                              end_date.strftime("%Y-%m-%d"), verbose=False)
                    
                    if df is not None and not df.empty:
                        latest = df.iloc[-1]
                        prev = df.iloc[-2] if len(df) > 1 else latest
                        
                        change = (latest['close'] - prev['close']) / prev['close'] * 100
                        index_performance.append(f"{symbol}: {latest['close']:.2f} ({change:+.2f}%)")
                except:
                    continue
            
            # 获取相关新闻
            news_summary = ""
            try:
                news_df = get_us_stock_news(limit=10, verbose=False)
                if news_df is not None and not news_df.empty:
                    top_news = news_df.head(3)
                    news_items = []
                    for _, row in top_news.iterrows():
                        title = str(row.get('title', ''))[:100]
                        news_items.append(f"• {title}")
                    news_summary = "\n".join(news_items)
            except:
                pass
            
            # 组装洞察报告
            content = f"""美股市场洞察报告 - {trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}

📊 主要指数表现：
{chr(10).join(index_performance) if index_performance else "数据暂不可用"}

📰 市场要闻：
{news_summary if news_summary else "暂无重要新闻"}

💡 市场观察：
基于当前指数表现和新闻动态，为投资决策提供参考依据。请结合基本面分析和技术指标进行综合判断。

⚠️  风险提示：
本报告仅供参考，不构成投资建议。投资有风险，决策需谨慎。
"""
            return content
            
        except Exception as e:
            logger.error(f"生成市场洞察失败: {e}")
            return f"市场洞察生成失败: {str(e)}"

    async def get_data(self, trigger_time: str) -> pd.DataFrame:
        try:
            # 检查缓存
            cached = self.get_data_cached(trigger_time)
            if cached is not None:
                return cached

            # 获取前一个交易日
            trade_date = get_previous_trading_date(trigger_time)
            
            # 生成市场洞察内容
            content = self._generate_market_insight(trade_date, trigger_time)
            
            # 构造返回数据
            data = [{
                'title': f"美股市场洞察 - {trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}",
                'content': content,
                'pub_time': trigger_time,
                'url': None
            }]
            
            df = pd.DataFrame(data)
            
            # 缓存结果
            self.save_data_cached(trigger_time, df)
            
            logger.info(f"生成美股市场洞察报告成功，交易日: {trade_date}")
            return df
            
        except Exception as e:
            logger.error(f"获取美股市场洞察失败: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    insight = USMarketInsight()
    print(asyncio.run(insight.get_data("2024-12-01 10:00:00")))
