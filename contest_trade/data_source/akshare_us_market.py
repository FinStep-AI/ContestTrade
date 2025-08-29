"""  
Akshare US Market data source
基于akshare获取美股市场数据，补充现有数据源
返回DataFrame列: ['title', 'content', 'pub_time', 'url']
"""
import pandas as pd
from datetime import datetime, timedelta
import traceback
from loguru import logger

import sys
from pathlib import Path

_CURRENT_FILE = Path(__file__).resolve()
_CONTEST_TRADE_DIR = _CURRENT_FILE.parents[1]
if str(_CONTEST_TRADE_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTEST_TRADE_DIR))

from data_source.data_source_base import DataSourceBase
from utils.date_utils import get_previous_trading_date

try:
    import akshare as ak
except ImportError:
    logger.warning("akshare not installed, akshare_us_market will not work")
    ak = None


class AkshareUSMarket(DataSourceBase):
    def __init__(self):
        super().__init__("akshare_us_market")
        
    def _get_us_stock_realtime(self, symbol: str) -> dict:
        """获取美股实时数据"""
        try:
            if ak is None:
                return None
            # 使用akshare获取美股实时数据
            df = ak.stock_us_spot_em()
            if df is not None and not df.empty:
                # 查找指定股票
                stock_data = df[df['代码'] == symbol]
                if not stock_data.empty:
                    row = stock_data.iloc[0]
                    return {
                        'symbol': symbol,
                        'name': row.get('名称', ''),
                        'current_price': row.get('最新价', 0),
                        'change': row.get('涨跌额', 0),
                        'change_percent': row.get('涨跌幅', 0),
                        'volume': row.get('成交量', 0),
                        'market_cap': row.get('总市值', 0)
                    }
            return None
        except Exception as e:
            logger.error(f"获取{symbol}实时数据失败: {e}")
            return None
    
    def _get_us_market_overview(self) -> dict:
        """获取美股市场概览"""
        try:
            if ak is None:
                return None
            
            # 获取美股实时行情
            df = ak.stock_us_spot_em()
            print(df)
            if df is None or df.empty:
                return None
                
            # 统计市场概况
            total_stocks = len(df)
            rising_stocks = len(df[df['涨跌幅'] > 0])
            falling_stocks = len(df[df['涨跌幅'] < 0])
            unchanged_stocks = total_stocks - rising_stocks - falling_stocks
            
            # 获取主要指数数据
            major_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
            major_data = []
            
            for symbol in major_stocks:
                stock_info = self._get_us_stock_realtime(symbol)
                if stock_info:
                    major_data.append(stock_info)
            
            return {
                'total_stocks': total_stocks,
                'rising_stocks': rising_stocks,
                'falling_stocks': falling_stocks,
                'unchanged_stocks': unchanged_stocks,
                'rising_ratio': round(rising_stocks / total_stocks * 100, 2) if total_stocks > 0 else 0,
                'major_stocks': major_data,
                'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"获取美股市场概览失败: {e}")
            return None
    
    def _get_us_stock_news(self, limit: int = 10) -> list:
        """获取美股相关新闻"""
        try:
            if ak is None:
                return []
            
            # 使用akshare获取财经新闻
            df = ak.news_cctv()
            if df is None or df.empty:
                return []
            
            news_list = []
            for _, row in df.head(limit).iterrows():
                news_item = {
                    'title': row.get('title', ''),
                    'content': row.get('content', ''),
                    'pub_time': row.get('time', ''),
                    'url': row.get('url', '')
                }
                news_list.append(news_item)
            
            return news_list
        except Exception as e:
            logger.error(f"获取美股新闻失败: {e}")
            return []
    
    def _build_market_analysis(self, trade_date_yyyymmdd: str) -> str:
        """构建市场分析报告"""
        try:
            # 获取市场概览
            market_overview = self._get_us_market_overview()
            if not market_overview:
                return "无法获取市场数据"
            
            # 构建分析内容
            analysis = f"""# Akshare美股市场分析报告

## 市场概览 ({market_overview['update_time']})
- 总股票数: {market_overview['total_stocks']}
- 上涨股票: {market_overview['rising_stocks']} ({market_overview['rising_ratio']}%)
- 下跌股票: {market_overview['falling_stocks']}
- 平盘股票: {market_overview['unchanged_stocks']}

## 主要股票表现
"""
            
            for stock in market_overview['major_stocks']:
                change_sign = "+" if stock['change'] >= 0 else ""
                analysis += f"- {stock['name']} ({stock['symbol']}): ${stock['current_price']:.2f} {change_sign}{stock['change']:.2f} ({stock['change_percent']:.2f}%)\n"
            
            # 添加市场趋势分析
            if market_overview['rising_ratio'] > 60:
                trend = "市场整体表现强劲，多数股票上涨"
            elif market_overview['rising_ratio'] > 40:
                trend = "市场表现平稳，涨跌相对均衡"
            else:
                trend = "市场承压，多数股票下跌"
            
            analysis += f"\n## 市场趋势\n{trend}\n"
            
            # 添加新闻摘要
            news_list = self._get_us_stock_news(5)
            if news_list:
                analysis += "\n## 相关新闻\n"
                for i, news in enumerate(news_list[:3], 1):
                    analysis += f"{i}. {news['title']}\n"
            
            return analysis
            
        except Exception as e:
            logger.error(f"构建市场分析失败: {e}")
            return f"市场分析生成失败: {str(e)}"
    
    def get_data(self, trigger_time: str) -> pd.DataFrame:
        """获取数据的主要接口"""
        try:
            if ak is None:
                logger.warning("akshare未安装，返回空数据")
                return pd.DataFrame(columns=['title', 'content', 'pub_time', 'url'])
            
            # 获取交易日
            trade_date_yyyymmdd = get_previous_trading_date(trigger_time)
            
            logger.info(f"Akshare美股数据源开始获取数据，交易日: {trade_date_yyyymmdd}")
            
            # 构建市场分析
            market_analysis = self._build_market_analysis(trade_date_yyyymmdd)
            
            # 获取新闻数据
            news_list = self._get_us_stock_news(10)
            
            # 构建返回数据
            data_list = []
            
            # 添加市场分析
            data_list.append({
                'title': f'Akshare美股市场分析 - {trade_date_yyyymmdd}',
                'content': market_analysis,
                'pub_time': trigger_time,
                'url': 'akshare://us_market_analysis'
            })
            
            # 添加新闻数据
            for news in news_list:
                if news['title'] and news['content']:
                    data_list.append({
                        'title': f"[Akshare新闻] {news['title']}",
                        'content': news['content'][:1000],  # 限制内容长度
                        'pub_time': news['pub_time'] or trigger_time,
                        'url': news['url'] or 'akshare://news'
                    })
            
            df = pd.DataFrame(data_list)
            logger.info(f"Akshare美股数据源获取完成，共{len(df)}条记录")
            
            return df
            
        except Exception as e:
            logger.error(f"Akshare美股数据源获取失败: {e}")
            logger.error(traceback.format_exc())
            # 返回空DataFrame而不是None，避免后续处理错误
            return pd.DataFrame(columns=['title', 'content', 'pub_time', 'url'])


if __name__ == "__main__":
    # 测试代码
    ds = AkshareUSMarket()
    test_time = "2024-12-19 09:00:00"
    result = ds.get_data(test_time)
    print(f"获取到{len(result)}条数据")
    if not result.empty:
        print(result[['title', 'pub_time']].head())
        print("\n第一条内容预览:")
        print(result.iloc[0]['content'][:200] + "...")