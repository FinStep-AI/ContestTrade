"""
US Price Market data source
美股价格市场数据源（替代原price_market.py）
整合美股K线数据、板块轮动、资金流向等，生成综合宏观市场分析
返回DataFrame列: ['title', 'content', 'pub_time', 'url']
""" 
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from data_source.data_source_base import DataSourceBase
from utils.fmp_utils import get_us_stock_price
from models.llm_model import GLOBAL_LLM
from loguru import logger
from utils.date_utils import get_previous_trading_date


class USPriceMarket(DataSourceBase):
    def __init__(self):
        super().__init__("us_price_market")

    def _get_sector_performance(self, trade_date_str: str) -> dict:
        """获取美股行业表现"""
        sector_etfs = {
            'XLK': '科技',
            'XLF': '金融', 
            'XLE': '能源',
            'XLV': '医疗',
            'XLI': '工业',
            'XLY': '消费(可选)',
            'XLP': '消费(必需)',
            'XLU': '公用事业',
            'XLB': '材料',
            'XLRE': '房地产',
            'XLC': '通信'
        }
        
        performance_data = {}
        
        for etf, sector_name in sector_etfs.items():
            try:
                # 获取近5日数据
                end_date = datetime.strptime(trade_date_str, "%Y%m%d")
                start_date = end_date - timedelta(days=7)
                
                df = get_us_stock_price(etf, from_date=start_date.strftime("%Y-%m-%d"), 
                                       to_date=end_date.strftime("%Y-%m-%d"), verbose=False)
                
                if df is not None and not df.empty and len(df) >= 2:
                    latest = df.iloc[-1]
                    prev = df.iloc[-2]
                    
                    change_pct = (latest['close'] - prev['close']) / prev['close'] * 100
                    volume_ratio = latest['volume'] / df['volume'].mean() if df['volume'].mean() > 0 else 1
                    
                    performance_data[sector_name] = {
                        'symbol': etf,
                        'change_pct': change_pct,
                        'close': latest['close'],
                        'volume_ratio': volume_ratio
                    }
                    
            except Exception as e:
                logger.warning(f"获取{etf}({sector_name})数据失败: {e}")
                continue
                
        return performance_data

    def _get_market_breadth(self, trade_date_str: str) -> dict:
        """获取市场广度数据"""
        try:
            # 主要指数
            indices = ['SPY', 'QQQ', 'DIA', 'IWM']
            index_data = {}
            
            for symbol in indices:
                try:
                    end_date = datetime.strptime(trade_date_str, "%Y%m%d")
                    start_date = end_date - timedelta(days=7)
                    
                    df = get_us_stock_price(symbol, from_date=start_date.strftime("%Y-%m-%d"), 
                                           to_date=end_date.strftime("%Y-%m-%d"), verbose=False)
                    
                    if df is not None and not df.empty and len(df) >= 2:
                        latest = df.iloc[-1]
                        prev = df.iloc[-2]
                        
                        change_pct = (latest['close'] - prev['close']) / prev['close'] * 100
                        index_data[symbol] = {
                            'change_pct': change_pct,
                            'close': latest['close'],
                            'volume': latest['volume']
                        }
                        
                except Exception:
                    continue
                    
            return index_data
            
        except Exception as e:
            logger.error(f"获取市场广度数据失败: {e}")
            return {}

    async def _generate_llm_summary(self, trade_date: str) -> dict:
        """使用LLM生成市场分析摘要"""
        try:
            # 获取数据
            sector_data = self._get_sector_performance(trade_date)
            market_data = self._get_market_breadth(trade_date)
            
            # 构造分析内容
            analysis_content = f"""美股市场数据分析 - {trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}

【主要指数表现】
"""
            
            for symbol, data in market_data.items():
                analysis_content += f"{symbol}: {data['close']:.2f} ({data['change_pct']:+.2f}%)\n"
            
            analysis_content += "\n【行业板块表现】\n"
            
            # 按涨跌幅排序
            sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1]['change_pct'], reverse=True)
            
            for sector_name, data in sorted_sectors:
                analysis_content += f"{sector_name}({data['symbol']}): {data['close']:.2f} ({data['change_pct']:+.2f}%)\n"
            
            analysis_content += "\n【资金流向分析】\n"
            
            # 分析成交量 TODO：FMP没有volume_ratio，后面需要补充
            high_volume_sectors = [name for name, data in sector_data.items() if data['volume_ratio'] > 1.2]
            if high_volume_sectors:
                analysis_content += f"成交量活跃板块: {', '.join(high_volume_sectors)}\n"
            
            # LLM分析
            prompt = f"""请基于以下美股市场数据，生成简洁的投资洞察分析（200字以内）：

{analysis_content}

要求：
1. 分析主要指数和板块的表现特点
2. 识别资金流向和热点板块
3. 提供简要的市场判断和风险提示
4. 语言简洁专业"""

            try:
                llm_response = await GLOBAL_LLM.a_run([{"role": "user", "content": prompt}], verbose=False)
                llm_summary = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            except Exception as e:
                logger.warning(f"LLM分析失败: {e}")
                llm_summary = "LLM分析暂不可用，请参考上述数据进行判断。"
            
            final_content = analysis_content + f"\n【AI市场洞察】\n{llm_summary}"
            
            return {
                "llm_summary": final_content,
                "raw_data": {
                    "sector_performance": sector_data,
                    "market_breadth": market_data
                }
            }
            
        except Exception as e:
            logger.error(f"生成LLM分析失败: {e}")
            return {
                "llm_summary": f"市场数据分析生成失败: {str(e)}",
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
            logger.info(f"获取 {trade_date} 的美股价格市场数据")

            # 生成LLM分析
            llm_summary_dict = await self._generate_llm_summary(trade_date)
            
            # 构造返回数据
            data = [{
                "title": f"{trade_date}:美股市场宏观数据汇总",
                "content": llm_summary_dict["llm_summary"],
                "pub_time": trigger_time,
                "url": None
            }]
            
            df = pd.DataFrame(data)
            
            # 检查LLM摘要中是否包含具体数据
            llm_summary = llm_summary_dict["llm_summary"]
            has_market_data = False
            has_sector_data = False
            
            # 检查【主要指数表现】部分是否有具体数据
            if "【主要指数表现】" in llm_summary:
                market_section_start = llm_summary.find("【主要指数表现】")
                market_section_end = llm_summary.find("【行业板块表现】")
                if market_section_end == -1:
                    market_section_end = llm_summary.find("【资金流向分析】")
                if market_section_end == -1:
                    market_section_end = len(llm_summary)
                
                market_section = llm_summary[market_section_start:market_section_end]
                # 检查是否包含具体的指数数据（包含数字和百分号）
                if any(char.isdigit() for char in market_section) and "%" in market_section:
                    has_market_data = True
            
            # 检查【行业板块表现】部分是否有具体数据
            if "【行业板块表现】" in llm_summary:
                sector_section_start = llm_summary.find("【行业板块表现】")
                sector_section_end = llm_summary.find("【资金流向分析】")
                if sector_section_end == -1:
                    sector_section_end = len(llm_summary)
                
                sector_section = llm_summary[sector_section_start:sector_section_end]
                # 检查是否包含具体的板块数据（包含数字和百分号）
                if any(char.isdigit() for char in sector_section) and "%" in sector_section:
                    has_sector_data = True
            
            # 只有当两个部分都有具体数据时才缓存
            if has_market_data and has_sector_data:
                logger.info(f"数据完整，缓存 {trade_date} 的美股价格市场数据")
                self.save_data_cached(trigger_time, df)
            else:
                logger.warning(f"数据不完整，跳过缓存 {trade_date} 的美股价格市场数据 (市场数据: {has_market_data}, 板块数据: {has_sector_data})")
            
            return df
                
        except Exception as e:
            logger.error(f"获取美股价格市场数据失败: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    price_market = USPriceMarket()
    print(asyncio.run(price_market.get_data("2024-12-01 15:00:00")))
