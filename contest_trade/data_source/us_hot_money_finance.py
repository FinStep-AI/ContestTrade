"""美股热钱数据源
获取美股异动股票、机构资金流向、期权活动、内幕交易等热钱相关数据
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
import yfinance as yf
import requests
from typing import Dict, List, Optional
import json

class USHotMoneyFinance(DataSourceBase):
    def __init__(self):
        super().__init__("us_hot_money_finance")
        
        # 美股主要指数成分股（用于筛选活跃股票）
        self.major_indices = {
            'SP500': '^GSPC',
            'NASDAQ': '^IXIC', 
            'DOW': '^DJI'
        }
        
        # 热门股票池（科技股、成长股等）
        self.hot_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'LYFT',
            'COIN', 'PLTR', 'SNOW', 'ZM', 'DOCU', 'ROKU', 'SQ', 'SHOP',
            'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI', 'DIDI', 'BIDU'
        ]
        
    async def get_data(self, trigger_time: str) -> pd.DataFrame:
        try:
            df = self.get_data_cached(trigger_time)
            if df is not None:
                return df
            
            logger.info(f"获取美股热钱数据: {trigger_time}")
            
            # 获取热钱数据
            hot_money_data = await self.fetch_hot_money_data(trigger_time)
            
            if not hot_money_data:
                logger.warning("未获取到美股热钱数据")
                return pd.DataFrame()
            
            # 使用LLM分析热钱数据
            llm_summary = await self.get_llm_summary(hot_money_data, trigger_time)
            
            # 构建返回数据
            data = [{
                "title": f"{trigger_time.split(' ')[0]}:美股热钱流向分析",
                "content": llm_summary,
                "pub_time": trigger_time,
                "url": None
            }]
            
            df = pd.DataFrame(data)
            self.save_data_cached(trigger_time, df)
            return df
                
        except Exception as e:
            logger.error(f"获取美股热钱数据失败: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    async def fetch_hot_money_data(self, trigger_time: str) -> dict:
        """获取热钱数据"""
        target_date = datetime.strptime(trigger_time.split(' ')[0], '%Y-%m-%d')
        
        hot_money_data = {
            'unusual_volume': [],
            'price_movers': [],
            'sector_flows': {},
            'market_sentiment': {}
        }
        
        try:
            # 获取异常成交量股票
            unusual_volume = await self.get_unusual_volume_stocks(target_date)
            hot_money_data['unusual_volume'] = unusual_volume
            
            # 获取价格异动股票
            price_movers = await self.get_price_movers(target_date)
            hot_money_data['price_movers'] = price_movers
            
            # 获取板块资金流向
            sector_flows = await self.get_sector_flows(target_date)
            hot_money_data['sector_flows'] = sector_flows
            
            # 获取市场情绪指标
            market_sentiment = await self.get_market_sentiment(target_date)
            hot_money_data['market_sentiment'] = market_sentiment
            
            return hot_money_data
            
        except Exception as e:
            logger.error(f"获取热钱数据失败: {e}")
            return hot_money_data
    
    async def get_unusual_volume_stocks(self, target_date: datetime) -> List[Dict]:
        """获取异常成交量股票"""
        unusual_stocks = []
        
        try:
            # 获取热门股票的成交量数据
            for symbol in self.hot_stocks[:20]:  # 限制数量避免API限制
                try:
                    ticker = yf.Ticker(symbol)
                    
                    # 获取最近5天的数据
                    end_date = target_date + timedelta(days=1)
                    start_date = target_date - timedelta(days=5)
                    
                    hist = ticker.history(start=start_date, end=end_date)
                    
                    if len(hist) < 2:
                        continue
                    
                    # 计算成交量异常
                    recent_volume = hist['Volume'].iloc[-1] if len(hist) > 0 else 0
                    avg_volume = hist['Volume'].iloc[:-1].mean() if len(hist) > 1 else recent_volume
                    
                    if avg_volume > 0:
                        volume_ratio = recent_volume / avg_volume
                        
                        # 成交量异常（超过平均值2倍）
                        if volume_ratio > 2.0:
                            price_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0
                            
                            unusual_stocks.append({
                                'symbol': symbol,
                                'volume_ratio': round(volume_ratio, 2),
                                'recent_volume': int(recent_volume),
                                'avg_volume': int(avg_volume),
                                'price_change': round(price_change, 2),
                                'close_price': round(hist['Close'].iloc[-1], 2)
                            })
                    
                    # 避免API限制
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"获取 {symbol} 成交量数据失败: {e}")
                    continue
            
            # 按成交量异常比例排序
            unusual_stocks.sort(key=lambda x: x['volume_ratio'], reverse=True)
            
            logger.info(f"发现 {len(unusual_stocks)} 只异常成交量股票")
            return unusual_stocks[:10]  # 返回前10只
            
        except Exception as e:
            logger.error(f"获取异常成交量股票失败: {e}")
            return []
    
    async def get_price_movers(self, target_date: datetime) -> Dict[str, List]:
        """获取价格异动股票"""
        movers = {
            'gainers': [],
            'losers': []
        }
        
        try:
            for symbol in self.hot_stocks[:30]:  # 扩大范围
                try:
                    ticker = yf.Ticker(symbol)
                    
                    # 获取最近2天的数据
                    end_date = target_date + timedelta(days=1)
                    start_date = target_date - timedelta(days=2)
                    
                    hist = ticker.history(start=start_date, end=end_date)
                    
                    if len(hist) < 2:
                        continue
                    
                    # 计算价格变化
                    price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100
                    volume = hist['Volume'].iloc[-1]
                    
                    stock_info = {
                        'symbol': symbol,
                        'price_change': round(price_change, 2),
                        'close_price': round(hist['Close'].iloc[-1], 2),
                        'volume': int(volume),
                        'high': round(hist['High'].iloc[-1], 2),
                        'low': round(hist['Low'].iloc[-1], 2)
                    }
                    
                    # 分类涨跌幅较大的股票
                    if price_change > 5:  # 涨幅超过5%
                        movers['gainers'].append(stock_info)
                    elif price_change < -5:  # 跌幅超过5%
                        movers['losers'].append(stock_info)
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"获取 {symbol} 价格数据失败: {e}")
                    continue
            
            # 排序
            movers['gainers'].sort(key=lambda x: x['price_change'], reverse=True)
            movers['losers'].sort(key=lambda x: x['price_change'])
            
            # 限制数量
            movers['gainers'] = movers['gainers'][:10]
            movers['losers'] = movers['losers'][:10]
            
            logger.info(f"发现 {len(movers['gainers'])} 只大涨股票, {len(movers['losers'])} 只大跌股票")
            return movers
            
        except Exception as e:
            logger.error(f"获取价格异动股票失败: {e}")
            return movers
    
    async def get_sector_flows(self, target_date: datetime) -> Dict:
        """获取板块资金流向（简化版）"""
        sector_flows = {}
        
        try:
            # 主要板块ETF
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV', 
                'Financial': 'XLF',
                'Consumer': 'XLY',
                'Energy': 'XLE',
                'Utilities': 'XLU',
                'Materials': 'XLB',
                'Industrial': 'XLI',
                'Real Estate': 'XLRE'
            }
            
            for sector, etf_symbol in sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf_symbol)
                    
                    # 获取最近2天数据
                    end_date = target_date + timedelta(days=1)
                    start_date = target_date - timedelta(days=2)
                    
                    hist = ticker.history(start=start_date, end=end_date)
                    
                    if len(hist) >= 2:
                        price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100
                        volume_change = (hist['Volume'].iloc[-1] - hist['Volume'].iloc[-2]) / hist['Volume'].iloc[-2] * 100 if hist['Volume'].iloc[-2] > 0 else 0
                        
                        sector_flows[sector] = {
                            'price_change': round(price_change, 2),
                            'volume_change': round(volume_change, 2),
                            'volume': int(hist['Volume'].iloc[-1])
                        }
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"获取 {sector} 板块数据失败: {e}")
                    continue
            
            logger.info(f"获取到 {len(sector_flows)} 个板块的资金流向数据")
            return sector_flows
            
        except Exception as e:
            logger.error(f"获取板块资金流向失败: {e}")
            return {}
    
    async def get_market_sentiment(self, target_date: datetime) -> Dict:
        """获取市场情绪指标"""
        sentiment = {}
        
        try:
            # 获取VIX恐慌指数
            vix_ticker = yf.Ticker('^VIX')
            end_date = target_date + timedelta(days=1)
            start_date = target_date - timedelta(days=2)
            
            vix_hist = vix_ticker.history(start=start_date, end=end_date)
            
            if len(vix_hist) >= 1:
                vix_level = vix_hist['Close'].iloc[-1]
                vix_change = (vix_hist['Close'].iloc[-1] - vix_hist['Close'].iloc[-2]) / vix_hist['Close'].iloc[-2] * 100 if len(vix_hist) >= 2 else 0
                
                sentiment['vix'] = {
                    'level': round(vix_level, 2),
                    'change': round(vix_change, 2),
                    'interpretation': self._interpret_vix(vix_level)
                }
            
            # 获取主要指数表现
            indices_performance = {}
            for name, symbol in self.major_indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date)
                    
                    if len(hist) >= 2:
                        change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100
                        indices_performance[name] = round(change, 2)
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"获取 {name} 指数数据失败: {e}")
                    continue
            
            sentiment['indices'] = indices_performance
            
            logger.info("获取市场情绪指标完成")
            return sentiment
            
        except Exception as e:
            logger.error(f"获取市场情绪指标失败: {e}")
            return {}
    
    def _interpret_vix(self, vix_level: float) -> str:
        """解释VIX水平"""
        if vix_level < 15:
            return "市场情绪乐观，波动率较低"
        elif vix_level < 25:
            return "市场情绪正常，波动率适中"
        elif vix_level < 35:
            return "市场情绪谨慎，波动率偏高"
        else:
            return "市场情绪恐慌，波动率极高"
    
    async def get_llm_summary(self, hot_money_data: dict, trigger_time: str) -> str:
        """使用LLM生成热钱分析报告"""
        try:
            if not hot_money_data:
                return "未获取到美股热钱数据"
            
            # 构建分析文本
            analysis_text = self._construct_analysis_text(hot_money_data, trigger_time)
            
            # LLM分析提示
            prompt = f"""请基于以下美股热钱流向数据，生成一份专业的资金流向分析报告：

{analysis_text}

请从以下几个维度进行分析：
1. 异常成交量股票分析 - 识别资金集中流入的个股
2. 价格异动分析 - 分析大涨大跌股票的资金推动因素
3. 板块资金流向 - 识别热门和冷门板块
4. 市场情绪判断 - 基于VIX和指数表现判断整体情绪
5. 投资策略建议 - 基于资金流向提供操作建议

要求：
- 分析要专业客观，突出资金流向特征
- 重点关注异常数据和趋势变化
- 提供具体的数据支撑和逻辑推理
- 给出明确的投资方向建议
- 字数控制在800-1200字"""

            try:
                llm_response = await GLOBAL_LLM.acall(prompt)
                return llm_response.strip()
            except Exception as e:
                logger.error(f"LLM分析失败: {e}")
                return f"LLM分析服务不可用。热钱数据概要：\n{analysis_text[:1000]}..."
                
        except Exception as e:
            logger.error(f"生成热钱分析失败: {e}")
            return f"热钱分析生成失败: {str(e)}"
    
    def _construct_analysis_text(self, hot_money_data: dict, trigger_time: str) -> str:
        """构建用于LLM分析的热钱数据文本"""
        lines = [f"=== {trigger_time.split(' ')[0]} 美股热钱流向分析 ==="]
        
        # 异常成交量股票
        unusual_volume = hot_money_data.get('unusual_volume', [])
        if unusual_volume:
            lines.append("\n【异常成交量股票】")
            for i, stock in enumerate(unusual_volume[:5], 1):
                lines.append(f"{i}. {stock['symbol']}: 成交量放大{stock['volume_ratio']}倍, 涨跌幅{stock['price_change']}%, 价格${stock['close_price']}")
        
        # 价格异动股票
        price_movers = hot_money_data.get('price_movers', {})
        if price_movers.get('gainers'):
            lines.append("\n【大涨股票】")
            for i, stock in enumerate(price_movers['gainers'][:5], 1):
                lines.append(f"{i}. {stock['symbol']}: +{stock['price_change']}%, 价格${stock['close_price']}, 成交量{stock['volume']:,}")
        
        if price_movers.get('losers'):
            lines.append("\n【大跌股票】")
            for i, stock in enumerate(price_movers['losers'][:5], 1):
                lines.append(f"{i}. {stock['symbol']}: {stock['price_change']}%, 价格${stock['close_price']}, 成交量{stock['volume']:,}")
        
        # 板块资金流向
        sector_flows = hot_money_data.get('sector_flows', {})
        if sector_flows:
            lines.append("\n【板块资金流向】")
            # 按价格变化排序
            sorted_sectors = sorted(sector_flows.items(), key=lambda x: x[1]['price_change'], reverse=True)
            for sector, data in sorted_sectors:
                lines.append(f"{sector}: {data['price_change']:+.2f}%, 成交量变化{data['volume_change']:+.2f}%")
        
        # 市场情绪
        market_sentiment = hot_money_data.get('market_sentiment', {})
        if market_sentiment:
            lines.append("\n【市场情绪指标】")
            
            vix_data = market_sentiment.get('vix', {})
            if vix_data:
                lines.append(f"VIX恐慌指数: {vix_data['level']} ({vix_data['change']:+.2f}%) - {vix_data['interpretation']}")
            
            indices = market_sentiment.get('indices', {})
            if indices:
                lines.append("主要指数表现:")
                for index, change in indices.items():
                    lines.append(f"  {index}: {change:+.2f}%")
        
        return "\n".join(lines)

if __name__ == "__main__":
    us_hot_money = USHotMoneyFinance()
    df = asyncio.run(us_hot_money.get_data("2024-01-19 09:00:00"))
    print(df.head())
    if len(df) > 0:
        print("美股热钱分析内容:")
        print(df.content.values[0])