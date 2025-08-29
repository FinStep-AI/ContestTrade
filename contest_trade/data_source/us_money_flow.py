"""
US Money Flow data source
美股资金流向数据源（替代hot_money.py）
基于FMP数据分析美股资金流向、机构持仓变化、期权流动等
返回DataFrame列: ['title', 'content', 'pub_time', 'url']
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
from utils.yahoo_utils import get_yahoo_stock_price
from models.llm_model import GLOBAL_LLM
from loguru import logger
from utils.date_utils import get_previous_trading_date


class USMoneyFlow(DataSourceBase):
    def __init__(self):
        super().__init__("us_money_flow")

    def _get_etf_flows(self, trade_date_str: str) -> dict:
        """获取主要ETF资金流向"""
        major_etfs = {
            'SPY': 'S&P500',
            'QQQ': 'NASDAQ科技',
            'IWM': '小盘股',
            'EFA': '发达市场',
            'VTI': '全市场',
            'GLD': '黄金',
            'TLT': '长期国债',
            'HYG': '高收益债'
        }
        
        flow_data = {}
        
        for etf, description in major_etfs.items():
            try:
                # 获取近5日数据计算资金流向
                end_date = datetime.strptime(trade_date_str, "%Y%m%d")
                start_date = end_date - timedelta(days=7)
                
                df = get_yahoo_stock_price(etf, start_date.strftime("%Y-%m-%d"), 
                                          end_date.strftime("%Y-%m-%d"), verbose=False)
                
                if df is not None and not df.empty and len(df) >= 2:
                    # 计算价格变化和成交量
                    latest = df.iloc[-1]
                    prev = df.iloc[-2]
                    avg_volume = df['volume'].mean()
                    
                    price_change = (latest['close'] - prev['close']) / prev['close'] * 100
                    volume_ratio = latest['volume'] / avg_volume if avg_volume > 0 else 1
                    
                    # 估算资金流向（简化计算）
                    estimated_flow = latest['volume'] * latest['close'] * (1 if price_change > 0 else -1)
                    
                    flow_data[etf] = {
                        'description': description,
                        'price_change': price_change,
                        'volume_ratio': volume_ratio,
                        'estimated_flow': estimated_flow / 1_000_000,  # 转换为百万美元
                        'volume': latest['volume']
                    }
                    
            except Exception as e:
                logger.warning(f"获取{etf}流向数据失败: {e}")
                continue
                
        return flow_data

    def _get_sector_rotation(self, trade_date_str: str) -> dict:
        """分析板块轮动"""
        sector_etfs = {
            'XLK': '科技',
            'XLF': '金融',
            'XLE': '能源', 
            'XLV': '医疗',
            'XLI': '工业',
            'XLY': '消费(可选)',
            'XLP': '消费(必需)',
            'XLU': '公用事业'
        }
        
        rotation_data = {}
        
        for etf, sector in sector_etfs.items():
            try:
                end_date = datetime.strptime(trade_date_str, "%Y%m%d")
                start_date = end_date - timedelta(days=7)
                
                df = get_yahoo_stock_price(etf, start_date.strftime("%Y-%m-%d"), 
                                          end_date.strftime("%Y-%m-%d"), verbose=False)
                
                if df is not None and not df.empty and len(df) >= 2:
                    latest = df.iloc[-1]
                    prev = df.iloc[-2]
                    
                    price_change = (latest['close'] - prev['close']) / prev['close'] * 100
                    avg_volume = df['volume'].mean()
                    volume_surge = latest['volume'] / avg_volume if avg_volume > 0 else 1
                    
                    rotation_data[sector] = {
                        'etf': etf,
                        'price_change': price_change,
                        'volume_surge': volume_surge,
                        'momentum_score': price_change * volume_surge  # 简单动量评分
                    }
                    
            except Exception:
                continue
                
        return rotation_data

    def _analyze_institutional_sentiment(self) -> dict:
        """分析机构情绪（基于VIX、期权等）"""
        try:
            # 获取VIX数据（恐慌指数）
            vix_data = {}
            try:
                # 注意：FMP可能需要付费才能获取VIX数据，这里使用SPY作为替代
                end_date = datetime.now()
                start_date = end_date - timedelta(days=5)
                
                # 使用SPY的波动率作为市场情绪指标
                spy_df = get_yahoo_stock_price('SPY', start_date.strftime("%Y-%m-%d"), 
                                              end_date.strftime("%Y-%m-%d"), verbose=False)
                
                if spy_df is not None and not spy_df.empty:
                    # 计算近期波动率
                    spy_df['returns'] = spy_df['close'].pct_change()
                    volatility = spy_df['returns'].std() * (252 ** 0.5) * 100  # 年化波动率
                    
                    latest_change = spy_df['close'].iloc[-1] / spy_df['close'].iloc[-2] - 1
                    
                    vix_data = {
                        'implied_volatility': volatility,
                        'market_direction': '上涨' if latest_change > 0 else '下跌',
                        'sentiment': '谨慎' if volatility > 20 else '乐观'
                    }
                    
            except Exception:
                vix_data = {'implied_volatility': None, 'market_direction': '未知', 'sentiment': '中性'}
                
            return vix_data
            
        except Exception as e:
            logger.error(f"分析机构情绪失败: {e}")
            return {}

    async def _generate_flow_analysis(self, trade_date: str) -> dict:
        """生成资金流向分析"""
        try:
            # 获取各类数据
            etf_flows = self._get_etf_flows(trade_date)
            sector_rotation = self._get_sector_rotation(trade_date)
            institutional_sentiment = self._analyze_institutional_sentiment()
            
            # 构造分析内容
            analysis_content = f"""美股资金流向分析 - {trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}

【主要ETF资金流向】
"""
            
            # ETF流向分析
            sorted_etfs = sorted(etf_flows.items(), key=lambda x: x[1]['estimated_flow'], reverse=True)
            for etf, data in sorted_etfs[:8]:  # 显示前8个
                flow_status = "流入" if data['estimated_flow'] > 0 else "流出"
                analysis_content += f"{etf}({data['description']}): {abs(data['estimated_flow']):.1f}M {flow_status} ({data['price_change']:+.2f}%)\n"
            
            analysis_content += "\n【板块轮动分析】\n"
            
            # 板块轮动
            sorted_sectors = sorted(sector_rotation.items(), key=lambda x: x[1]['momentum_score'], reverse=True)
            hot_sectors = [name for name, data in sorted_sectors[:3]]
            cold_sectors = [name for name, data in sorted_sectors[-3:]]
            
            analysis_content += f"热门板块: {', '.join(hot_sectors)}\n"
            analysis_content += f"冷门板块: {', '.join(cold_sectors)}\n"
            
            analysis_content += "\n【市场情绪指标】\n"
            if institutional_sentiment:
                if institutional_sentiment.get('implied_volatility'):
                    analysis_content += f"市场波动率: {institutional_sentiment['implied_volatility']:.1f}%\n"
                analysis_content += f"市场方向: {institutional_sentiment.get('market_direction', '未知')}\n"
                analysis_content += f"整体情绪: {institutional_sentiment.get('sentiment', '中性')}\n"
            
            # LLM分析
            prompt = f"""请基于以下美股资金流向数据，生成投资洞察（150字以内）：

{analysis_content}

要求：
1. 分析资金流向的主要特征
2. 识别板块轮动趋势
3. 评估市场风险偏好
4. 提供简要投资建议"""

            try:
                llm_response = await GLOBAL_LLM.a_run([{"role": "user", "content": prompt}], verbose=False)
                llm_summary = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            except Exception as e:
                logger.warning(f"LLM分析失败: {e}")
                llm_summary = "AI分析暂不可用，请参考上述数据自行判断。"
            
            final_content = analysis_content + f"\n【智能分析】\n{llm_summary}"
            
            return {
                "llm_summary": final_content,
                "raw_data": {
                    "etf_flows": etf_flows,
                    "sector_rotation": sector_rotation,
                    "institutional_sentiment": institutional_sentiment
                }
            }
            
        except Exception as e:
            logger.error(f"生成资金流向分析失败: {e}")
            return {
                "llm_summary": f"资金流向分析生成失败: {str(e)}",
                "raw_data": {}
            }

    async def get_data(self, trigger_time: str) -> pd.DataFrame:
        try:
            # 检查缓存
            df = self.get_data_cached(trigger_time)
            if df is not None:
                return df
            
            # 获取交易日
            trade_date = get_previous_trading_date(trigger_time)
            logger.info(f"获取 {trade_date} 的美股资金流向数据")

            # 生成流向分析
            flow_analysis = await self._generate_flow_analysis(trade_date)
            
            # 构造返回数据
            data = [{
                "title": f"{trade_date}:美股资金流向分析",
                "content": flow_analysis["llm_summary"],
                "pub_time": trigger_time,
                "url": None
            }]
            
            df = pd.DataFrame(data)
            
            # 缓存结果
            self.save_data_cached(trigger_time, df)
            return df
                
        except Exception as e:
            logger.error(f"获取美股资金流向数据失败: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    money_flow = USMoneyFlow()
    print(asyncio.run(money_flow.get_data("2024-12-01 15:00:00")))
