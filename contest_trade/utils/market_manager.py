"""
Market Manager: Get Market informations and symbols.

Current Core Function:

1. Maintain the symbol pools for each market.
2. Get the price of the symbol by given symbol name and datetime.
3. Check whether the trade action can be executed by given symbol name and datetime.
4. Calculate trading costs and slippage for different markets.
"""
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

import textwrap
import pandas as pd
import yaml
from functools import lru_cache
from tqdm import tqdm
from typing import List, Dict, Optional, Union
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
# 可选加载：仅在使用A股/港股功能时需要
try:
    from utils.tushare_utils import pro_cached
except Exception:
    pro_cached = None
from utils.yahoo_utils import get_yahoo_stock_price
try:
    from utils.fmp_utils import fmp_cached
except Exception:
    fmp_cached = None

class Market(Enum):
    """市场枚举"""
    A_ALL = "CN-Stock"    # A股全量
    A_ETF = "CN-ETF"    # A股ETF
    HK = "HK-Stock"          # H股/港股
    US = "US-Stock"          # 美股
    CSI300 = "CSI300"        # 沪深300
    CSI500 = "CSI500"        # 中证500
    CSI1000 = "CSI1000"      # 中证1000

@dataclass
class TradingCostConfig:
    """交易成本配置基类"""
    slippage_rate: float = 0.001  # 滑点率
    slippage_mode: str = "percentage"  # "percentage" 或 "fixed"
    slippage_fixed: float = 0.01  # 固定滑点金额

@dataclass
class AStockTradingConfig(TradingCostConfig):
    """A股交易成本配置"""
    commission_rate: float = 0.0003      # 佣金费率 万3
    commission_min: float = 5.0          # 最低佣金 5元
    stamp_tax_rate: float = 0.001        # 印花税 千1 (仅卖出)
    transfer_fee_rate: float = 0.00002   # 过户费 万0.2
    slippage_rate: float = 0.001         # 滑点率 0.1%
    min_shares: int = 100                # 最小交易单位 100股

@dataclass
class ETFTradingConfig(TradingCostConfig):
    """ETF交易成本配置"""
    commission_rate: float = 0.0003      # 佣金费率 万3
    commission_min: float = 5.0          # 最低佣金 5元
    stamp_tax_rate: float = 0.001        # 印花税 千1 (仅卖出)
    transfer_fee_rate: float = 0.00002   # 过户费 万0.2

@dataclass
class USStockTradingConfig(TradingCostConfig):
    """美股交易成本配置"""
    fee_type: str = "zero_commission"    # "per_trade", "per_share", "zero_commission"
    commission_per_trade: float = 0.0    # 每笔交易费用
    commission_per_share: float = 0.0    # 每股费用
    slippage_rate: float = 0.001         # 滑点率 0.1%
    slippage_mode: str = "percentage"    # "percentage" 或 "fixed"
    slippage_fixed: float = 0.01         # 固定滑点金额 ($0.01)
    min_shares: int = 1                  # 最小交易单位 1股

@dataclass
class HKStockTradingConfig(TradingCostConfig):
    """港股交易成本配置"""
    commission_rate: float = 0.0025      # 佣金费率 0.25%
    commission_min: float = 100.0        # 最低佣金 100港币
    stamp_duty_rate: float = 0.0013      # 印花税 0.13%
    trading_fee_rate: float = 0.00005    # 交易费 0.005%
    slippage_rate: float = 0.002         # 滑点率 0.2%
    min_shares: int = 100                # 最小交易单位 100股

@dataclass
class MarketManagerConfig:
    """市场管理器配置"""
    target_markets: List[str]
    custom_symbols: List[Dict]
    trading_configs: Dict[str, TradingCostConfig]

    @classmethod
    def from_config_file(cls, config_path: str = "config/market_config.yaml") -> "MarketManagerConfig":
        """从配置文件加载市场管理器配置"""
        config_file = Path(config_path)
        if not config_file.exists():
            config_file = PROJECT_ROOT / config_path
        
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 解析目标市场
        target_markets = config_data.get('target_markets', [])
        
        # 解析自定义符号
        custom_symbols_raw = config_data.get('custom_symbols', {})
        custom_symbols = []
        for market_name, symbols in custom_symbols_raw.items():
            if symbols:  # 确保符号列表不为空
                for symbol in symbols:
                    custom_symbols.append({
                        "market": market_name,
                        "symbol": symbol
                    })
        
        # 解析交易成本配置
        trading_costs_data = config_data.get('trading_costs', {})
        trading_configs = {}
        
        for market_name, cost_config in trading_costs_data.items():
            config_type = cost_config.get('type', 'unknown')
            
            if config_type == 'a_stock':
                trading_configs[market_name] = AStockTradingConfig(
                    commission_rate=cost_config.get('commission_rate', 0.0003),
                    commission_min=cost_config.get('commission_min', 5.0),
                    stamp_tax_rate=cost_config.get('stamp_tax_rate', 0.001),
                    transfer_fee_rate=cost_config.get('transfer_fee_rate', 0.00002),
                    slippage_rate=cost_config.get('slippage_rate', 0.001),
                    slippage_mode=cost_config.get('slippage_mode', 'percentage'),
                    slippage_fixed=cost_config.get('slippage_fixed', 0.01),
                    min_shares=cost_config.get('min_shares', 100)
                )
            elif config_type == 'us_stock':
                trading_configs[market_name] = USStockTradingConfig(
                    fee_type=cost_config.get('fee_type', 'zero_commission'),
                    commission_per_trade=cost_config.get('commission_per_trade', 0.0),
                    commission_per_share=cost_config.get('commission_per_share', 0.0),
                    slippage_rate=cost_config.get('slippage_rate', 0.001),
                    slippage_mode=cost_config.get('slippage_mode', 'percentage'),
                    slippage_fixed=cost_config.get('slippage_fixed', 0.01),
                    min_shares=cost_config.get('min_shares', 1)
                )
            elif config_type == 'hk_stock':
                trading_configs[market_name] = HKStockTradingConfig(
                    commission_rate=cost_config.get('commission_rate', 0.0025),
                    commission_min=cost_config.get('commission_min', 100.0),
                    stamp_duty_rate=cost_config.get('stamp_duty_rate', 0.0013),
                    trading_fee_rate=cost_config.get('trading_fee_rate', 0.00005),
                    slippage_rate=cost_config.get('slippage_rate', 0.002),
                    slippage_mode=cost_config.get('slippage_mode', 'percentage'),
                    slippage_fixed=cost_config.get('slippage_fixed', 0.02),
                    min_shares=cost_config.get('min_shares', 100)
                )
        
        return cls(
            target_markets=target_markets,
            custom_symbols=custom_symbols,
            trading_configs=trading_configs
        )


class MarketManager:
    """股票代码管理器"""
    
    def __init__(self, config: MarketManagerConfig):
        self.config = config
        self.cache_dir = Path(__file__).parent / "cache" / "market_manager"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理标准市场
        self.target_markets = []
        for market in config.target_markets:
            try:
                self.target_markets.append(Market(market))
            except ValueError:
                raise ValueError(f"Invalid market: {market}")
        
        # 处理自定义符号，按市场分组
        self.custom_symbols = config.custom_symbols
        self.custom_symbols_by_market = {}
        for symbol_config in config.custom_symbols:
            market = symbol_config.get("market")
            symbol = symbol_config.get("symbol")
            if market and symbol:
                if market not in self.custom_symbols_by_market:
                    self.custom_symbols_by_market[market] = []
                self.custom_symbols_by_market[market].append(symbol)
                

    def get_target_symbol_list(self, trigger_time: str, code_only: bool = True):
        """获取目标符号列表"""
        symbol_list = []
        for market in self.target_markets:
            symbols = self.get_market_symbols(market, trigger_time, full_market=True).to_dict(orient="records")
            for symbol in symbols:
                if code_only:
                    symbol_list.append(symbol["ts_code"])
                else:
                    symbol_list.append({
                        'market': market.value,
                        'symbol': symbol["ts_code"],
                        'name': symbol["name"]
                    })

        for market_name, symbols in self.custom_symbols_by_market.items():
            for symbol in symbols:
                if code_only:
                    symbol_list.append(symbol)
                else:
                    symbol_list.append({
                        'market': market_name,
                        'symbol': symbol,
                        'name': self.get_stock_name_by_code(symbol, market_name)
                    })
        return symbol_list


    def get_stock_name_by_code(self, symbol, market):
        """根据股票代码获取股票名称
        
        Args:
            symbol (str): 股票代码，A股格式如600519.SH，美股格式如AAPL
            market (str): 市场类型，CN-Stock或US-Stock
        
        Returns:
            str: 股票名称，获取失败时返回原symbol
        """
        
        if market == "CN-Stock":
            func_kwargs = {
                'ts_code': symbol,
                'fields': 'ts_code,name,area,industry,list_date'
            }
            df = pro_cached.run("stock_basic", func_kwargs=func_kwargs, verbose=False)
            
            if df is None or df.empty:
                return symbol  # 如果查询失败，返回原代码
                
            stock_name = df.iloc[0]['name']
            return stock_name
        elif market == "US-Stock":
            # 美股：symbol已经是标准格式如AAPL
            try:
                profile_data = fmp_cached.run(f'profile/{symbol}', {})
                if profile_data and len(profile_data) > 0:
                    company_name = profile_data[0].get('companyName', symbol)
                    return company_name
                else:
                    return symbol
            except Exception as e:
                return symbol
        else:
            raise ValueError(f"不支持的市场类型: {market}")



    def get_trading_config(self, market_name: str) -> TradingCostConfig:
        """获取指定市场的交易成本配置"""
        return self.config.trading_configs.get(market_name, TradingCostConfig())

    def get_target_symbol_context(self, trigger_time: str):
        """获取目标符号上下文"""
        prompt = textwrap.dedent("""
        You can invest in the following targets:

        {market_description}

        Please confirm that for each market, you can only select from the given available symbols in your investment decisions.
        """)

        market_descriptions = {
            Market.A_ALL: ("CN-Stock", "All symbols in Chinese mainland stock market. ~5000+ A-shares", ["000001.SZ", "600519.SH", "000858.SZ"]),
            Market.A_ETF: ("CN-ETF", "All symbols in Chinese mainland ETF market. ~300+ A-share ETFs", ["510300.SH", "159919.SZ", "512880.SH"]),
            Market.HK: ("HK-Stock", "All symbols in Hong Kong stock market. ~2000+ HK stocks", ["00700.HK", "09988.HK", "01299.HK"]),
            Market.US: ("US-Stock", "All symbols in US stock market. ~8000+ US stocks", ["AAPL", "MSFT", "GOOGL"]),
            Market.CSI300: ("CSI300", "沪深300指数成分股，包含沪深两市最具代表性的300只大盘蓝筹股。", []),
            Market.CSI500: ("CSI500", "中证500指数成分股，包含沪深两市最具代表性的500只中小盘股。", []),
            Market.CSI1000: ("CSI1000", "中证1000指数成分股，包含沪深两市最具代表性的1000只中小盘股。", []),
        }
        
        market_lines = []
        
        # 处理标准市场
        for market in self.target_markets:
            if market in market_descriptions:
                name, desc, examples = market_descriptions[market]
                
                # 如果有自定义符号，使用自定义符号列表
                if name in self.custom_symbols_by_market:
                    custom_symbols = self.custom_symbols_by_market[name]
                    available_symbols = f"[{', '.join(custom_symbols)}]"
                else:
                    # 使用默认的全市场描述
                    available_symbols = f"{desc}. Eg. {', '.join(examples)}"
                
                market_lines.append(f"market_name: {name}")
                market_lines.append(f"available_symbols: {available_symbols}")
                market_lines.append("")  # 空行分隔
        
        # 处理只有自定义符号但没有对应标准市场的情况
        for market_name, symbols in self.custom_symbols_by_market.items():
            if not any(m.value == market_name for m in self.target_markets):
                available_symbols = f"[{', '.join(symbols)}]"
                market_lines.append(f"market_name: {market_name}")
                market_lines.append(f"available_symbols: {available_symbols}")
                market_lines.append("")  # 空行分隔
        
        # 移除最后的空行
        if market_lines and market_lines[-1] == "":
            market_lines.pop()
        
        prompt = prompt.format(
            market_description=chr(10).join(market_lines)
        )
        return prompt

    def get_market_symbols(self, market: Union[Market, str], trigger_time: str, full_market: bool = False):
        """获取市场符号"""
        target_date = trigger_time.split(" ")[0].replace("-", "")
        
        # 转换为字符串格式
        if isinstance(market, Market):
            market_name = market.value
        else:
            market_name = market
        
        # 如果有自定义符号，只返回自定义符号
        if market_name in self.custom_symbols_by_market and not full_market:
            symbols_data = []
            for symbol in self.custom_symbols_by_market[market_name]:
                symbols_data.append({
                    'ts_code': symbol,
                    'name': symbol,  # 可以后续从数据源获取真实名称
                    'market': market_name
                })
            return pd.DataFrame(symbols_data)
        
        # 标准市场处理
        if isinstance(market, str):
            try:
                market = Market(market)
            except ValueError:
                raise ValueError(f"Unknown market: {market}")

        if market == Market.A_ALL:
            df = pro_cached.run(
                func_name="bak_basic", 
                func_kwargs={
                    "trade_date": target_date,
                    "fields": "ts_code,name"
                }
            )
        elif market == Market.CSI300:
            df = pro_cached.run(
                func_name="index_weight", 
                func_kwargs={
                    "index_code": "000300.SH",
                    "trade_date": "20250630",
                }
            )
            df['ts_code'] = df['con_code']
        elif market == Market.CSI500:
            df = pro_cached.run(
                func_name="index_weight", 
                func_kwargs={
                    "index_code": "000905.SH",
                    "trade_date": "20250630",
                }
            )
            df['ts_code'] = df['con_code']
        elif market == Market.CSI1000:
            df = pro_cached.run(
                func_name="index_weight", 
                func_kwargs={
                    "index_code": "000852.SH",
                    "trade_date": "20250630",
                }
            )
            df['ts_code'] = df['con_code']
        elif market == Market.A_ETF:
            df = pro_cached.run(
                func_name="fund_basic", 
                func_kwargs={
                    "market": "E",
                    "fields": "ts_code,name,list_date",
                }
            )
            df = df[df["list_date"] < target_date]
        elif market == Market.HK:
            df = pro_cached.run(
                func_name="hk_basic", 
                func_kwargs={
                    "fields": "ts_code,name,list_date"
                }
            )
            df = df[df["list_date"] < target_date]
        elif market == Market.US:
            # 优先使用FMP全量列表，失败时回退tushare
            try:
                stocks = fmp_cached.run('/stock/list', {})
                df = pd.DataFrame(stocks)
                # 仅保留主要交易所
                df = df[df['exchangeShortName'].isin(['NASDAQ', 'NYSE', 'AMEX'])]
                df = df[['symbol', 'name']].dropna()
                df = df.rename(columns={'symbol': 'ts_code'})
            except Exception:
                df_all = []
                for i in range(5):
                    df_part = pro_cached.run(
                        func_name="us_basic",
                        func_kwargs={
                            "fields": "ts_code,name,list_date,delist_date",
                            "offset": 5000 * i,
                            "limit": 5000
                        }
                    )
                    df_all.append(df_part)
                df = pd.concat(df_all)
                df = df[df["list_date"] < target_date]
                df = df[~(df["delist_date"] < target_date)]
                df = df.dropna(subset=["ts_code", "list_date"])
                df['name'] = df['ts_code']
        else:
            raise ValueError(f"Invalid market: {market}")
        return df

    def get_trade_date(self, market_name: str="CN-Stock", verbose: bool = False):
        if market_name in ["CN-Stock", "CSI300", "CSI500", "CSI1000"]:
            trade_date = pro_cached.run(
                func_name="trade_cal",
                func_kwargs={
                    "exchange": "SSE"
                }
            )
        elif market_name == "CN-ETF":
            trade_date = pro_cached.run(
                func_name="trade_cal",
                func_kwargs={
                    "exchange": "SSE"
                }
            )
        elif market_name == "HK-Stock":
            trade_date = pro_cached.run(
                func_name="hk_tradecal",
                func_kwargs={
                }
            )
        elif market_name == "US-Stock":
            # 优先使用Yahoo Finance的SPY价格序列构造交易日
            try:
                df = get_yahoo_stock_price('SPY', '1990-01-01', '2030-01-01', verbose=False)
                if df is not None and not df.empty:
                    trade_date = pd.DataFrame({'cal_date': df['date'].dt.strftime('%Y%m%d'), 'is_open': 1})
                else:
                    raise Exception("Yahoo Finance data unavailable")
            except Exception:
                if pro_cached:
                    trade_date = pro_cached.run(
                        func_name="us_tradecal",
                        func_kwargs={}
                    )
                else:
                    # 如果没有tushare，使用默认的工作日
                    from datetime import datetime, timedelta
                    start_date = datetime(1990, 1, 1)
                    end_date = datetime(2030, 12, 31)
                    dates = []
                    current = start_date
                    while current <= end_date:
                        # 排除周末 (Monday=0, Sunday=6)
                        if current.weekday() < 5:
                            dates.append(current.strftime('%Y%m%d'))
                        current += timedelta(days=1)
                    trade_date = pd.DataFrame({'cal_date': dates, 'is_open': 1})
        else:
            raise ValueError(f"Invalid market: {market_name}")
        trade_date = trade_date[trade_date["is_open"] == 1]["cal_date"].values.tolist()
        trade_date.sort()
        return trade_date

    def get_symbol_price(self, market_name: str, symbol: str, trigger_time: str, date_diff: int = 0):
        # get the open price of the symbol at given trigger_time
        triggle_trade_date = trigger_time.split(" ")[0].replace("-", "")
        trade_dates = self.get_trade_date(market_name)
        try:
            if date_diff == 0:
                assert triggle_trade_date in trade_dates
                target_trade_date = triggle_trade_date
            elif date_diff > 0:
                target_trade_date = [dt for dt in trade_dates if dt > triggle_trade_date][date_diff - 1]
            elif date_diff < 0:
                target_trade_date = [dt for dt in trade_dates if dt < triggle_trade_date][date_diff]
        except IndexError:
            raise ValueError(f"Invalid date_diff: {date_diff}")

        if market_name in ["CN-Stock", "CSI300", "CSI500", "CSI1000"]:
            df = pro_cached.run(
                func_name="daily",
                func_kwargs={"ts_code": symbol,"trade_date": target_trade_date}
            )
            limit_df = pro_cached.run(
                func_name="stk_limit",
                func_kwargs={
                    "trade_date": target_trade_date,
                }
            )
            limit_price = limit_df[limit_df['ts_code'] == symbol]['up_limit'].values[0]

            # calculate qfq price
            if target_trade_date < "20250630":
                # Calculate front-adjusted price using 20250630 as reference only support date before 20250630
                target_adj_data = pro_cached.run(
                    func_name='adj_factor', 
                    func_kwargs={
                        'ts_code': symbol,
                        'start_date': target_trade_date,
                        'end_date': '20250630'
                    }
                )
                target_adj_data = target_adj_data.sort_values(by="trade_date", ascending=True)
                qfq_factor = target_adj_data['adj_factor'].iloc[0] / target_adj_data['adj_factor'].iloc[-1]
                for price_fields in ["open", "high", "low", "close", "pre_close"]:
                    df[price_fields] = df[price_fields] * qfq_factor
            else:
                qfq_factor = 1.0


            if len(df) > 0:
                price_data = df.to_dict(orient="records")[0]
                price_data["limit_price"] = limit_price * qfq_factor
                return price_data
            else:
                return None
        elif market_name == "CN-ETF":
            return None
        elif market_name == "HK-Stock":
            return None
        elif market_name == "US-Stock":
            print(f"get_symbol_price: {market_name} {symbol} {target_trade_date}")
            
            try:
                # 获取指定日期的历史数据（使用FMP默认前复权价格，适用于回测）
                df = get_us_stock_price(symbol, target_trade_date, target_trade_date, 
                                      adjusted=True, adj_base_date=None, verbose=False)
                if not df.empty:
                    # 转换为与tushare格式兼容的字典
                    row = df.iloc[0]
                    price_data = {
                        'ts_code': symbol,
                        'trade_date': target_trade_date,
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'pre_close': row.get('adjClose', row['close']),  # 使用调整后收盘价
                        'change': row.get('change', 0),
                        'pct_chg': row.get('changePercent', 0),
                        'vol': row.get('volume', 0),
                        'amount': row.get('volume', 0) * row['close']  # 估算成交额
                    }
                    print(price_data)
                    return price_data
            except Exception as e:
                print(f"FMP数据获取失败，尝试tushare: {e}")
       
            return None
        else:
            raise ValueError(f"Invalid market: {market_name}")

    def get_symbol_history_price(self, market_name: str, symbol: str, start_date: str, end_date: str):
        if market_name in ["CN-Stock", "CN-ETF", "CSI300", "CSI500", "CSI1000"]:
            df = pro_cached.run(
                func_name="daily",
                func_kwargs={
                    "ts_code": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )
            return df
        elif market_name in ["CN-Index"]:
            df = pro_cached.run(
                func_name="index_daily",
                func_kwargs={
                    "ts_code": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )
            return df
        elif market_name == "US-Stock":
            df = get_yahoo_stock_price(symbol, start_date, end_date, verbose=False)
            return df
        else:
            raise ValueError(f"Invalid market: {market_name}")

    def accept_trade(self, symbol: str, action: str, trigger_time: str):
        # check if the trade is accepted by the market
        pass

    def is_market_trading(self, market_name: str, trigger_time: str):
        # check if the market is trading at given trigger_time  
        trigger_date = trigger_time.split(" ")[0].replace("-", "")
        trade_date = self.get_trade_date(market_name)
        if market_name in ["CN-Stock", "CN-ETF", "US-Stock", "CSI300", "CSI500", "CSI1000"]:
            return trigger_date in trade_date
        elif market_name == "HK-Stock":
            # Not supported yet
            return False
        else:
            raise ValueError(f"Invalid market: {market_name}")
        
    def is_available_symbol(self, market_name: str, symbol: str):
        # check if the symbol is custom symbol
        for symbol_config in self.custom_symbols:
            if symbol_config["market"] == market_name and symbol_config["symbol"] == symbol:
                return True
        
        # check if the symbol is in the market
        if market_name in ["CN-Stock", "CN-ETF", "US-Stock", "HK-Stock", "CSI300", "CSI500", "CSI1000"]:
            try:
                market_symbols = self.get_market_symbols(market_name, "2025-06-30 15:00:00")
                return symbol in market_symbols['ts_code'].values
            except Exception as e:
                print(f"Error checking symbol {symbol} in market {market_name}: {e}")
                return False
        else:
            raise ValueError(f"Invalid market: {market_name}")
        return False
    
    def get_trading_config(self, market_name: str) -> TradingCostConfig:
        """获取指定市场的交易成本配置"""
        return self.config.trading_configs.get(market_name, TradingCostConfig())

    def calculate_tradable_shares(self, market_name: str, target_amount: float, price: float) -> int:
        """计算可交易股数"""
        config = self.get_trading_config(market_name)
        
        if market_name in ["CN-Stock", "CN-ETF", "CSI300", "CSI500", "CSI1000"]:
            # A股：100股起，整手交易
            shares_float = target_amount / price
            return int(shares_float // 100) * 100
        elif market_name == "US-Stock":
            # 美股：1股起，任意股数
            return int(target_amount / price)
        elif market_name == "HK-Stock":
            # 港股：通常100股起
            shares_float = target_amount / price
            return int(shares_float // 100) * 100
        else:
            raise ValueError(f"不支持的市场类型: {market_name}")
    
    def apply_slippage(self, market_name: str, price: float, action: str, symbol: str) -> float:
        """应用滑点计算实际成交价格"""
        config = self.get_trading_config(market_name)
        
        if config.slippage_mode == "percentage":
            slippage = price * config.slippage_rate
        else:  # fixed
            slippage = config.slippage_fixed
        
        # 买入时价格上滑，卖出时价格下滑
        if action == "buy":
            return price + slippage
        else:  # sell
            return price - slippage
    
    def calculate_trading_costs(self, market_name: str, action: str, shares: int, 
                              price: float, symbol: str) -> Dict[str, float]:
        """计算交易成本"""
        config = self.get_trading_config(market_name)
        amount = shares * price
        
        if market_name in ["CN-Stock", "CN-ETF", "CSI300", "CSI500", "CSI1000"]:
            return self._calculate_a_stock_costs(config, action, shares, amount)
        elif market_name == "US-Stock":
            return self._calculate_us_stock_costs(config, action, shares, amount)
        elif market_name == "HK-Stock":
            return self._calculate_hk_stock_costs(config, action, shares, amount)
        else:
            raise ValueError(f"不支持的市场类型: {market_name}")
    
    def _calculate_a_stock_costs(self, config: AStockTradingConfig, action: str, 
                               shares: int, amount: float) -> Dict[str, float]:
        """计算A股交易成本"""
        # 佣金 (双向)
        commission = max(amount * config.commission_rate, config.commission_min)
        
        # 印花税 (仅卖出，ETF可能为0)
        stamp_tax = amount * config.stamp_tax_rate if action == "sell" else 0.0
        
        # 过户费 (双向)
        transfer_fee = amount * config.transfer_fee_rate
        
        total_cost = commission + stamp_tax + transfer_fee
        
        return {
            'commission': commission,
            'stamp_tax': stamp_tax,
            'transfer_fee': transfer_fee,
            'slippage_cost': 0.0,  # 滑点在价格中已体现
            'total_cost': total_cost
        }
    
    def _calculate_us_stock_costs(self, config: USStockTradingConfig, action: str, 
                                shares: int, amount: float) -> Dict[str, float]:
        """计算美股交易成本"""
        if config.fee_type == "per_trade":
            commission = config.commission_per_trade
        elif config.fee_type == "per_share":
            commission = shares * config.commission_per_share
        else:  # zero_commission
            commission = 0.0
        
        return {
            'commission': commission,
            'stamp_tax': 0.0,
            'transfer_fee': 0.0,
            'slippage_cost': 0.0,  # 滑点在价格中已体现
            'total_cost': commission
        }
    
    def _calculate_hk_stock_costs(self, config: HKStockTradingConfig, action: str, 
                                shares: int, amount: float) -> Dict[str, float]:
        """计算港股交易成本"""
        # 佣金
        commission = max(amount * config.commission_rate, config.commission_min)
        
        # 印花税
        stamp_duty = amount * config.stamp_duty_rate
        
        # 交易费
        trading_fee = amount * config.trading_fee_rate
        
        total_cost = commission + stamp_duty + trading_fee
        
        return {
            'commission': commission,
            'stamp_tax': stamp_duty,
            'transfer_fee': trading_fee,
            'slippage_cost': 0.0,  # 滑点在价格中已体现
            'total_cost': total_cost
        }
    
    def fix_symbol_code(self, market_name: str, symbol_name: str, symbol_code: str, verbose: bool = False):
        if market_name == "CN-Stock":
            stock_name2code, stock_code2name = self.get_stock_mapping(market_name)
            stock_act_code = stock_name2code.get(symbol_name, "None")
            if stock_act_code != symbol_code:
                if stock_act_code != "None":
                    if verbose:
                        print(f"set stock_code: {symbol_code} to {stock_act_code}")
                    symbol_code = stock_act_code
            stock_act_name = stock_code2name.get(symbol_code, "xxxxx")
            if stock_act_name in symbol_name and stock_act_name != symbol_name:
                if verbose:
                    print(f"set stock_name: {symbol_name} to {stock_act_name}")
                if stock_act_name != "xxxxx":
                    symbol_name = stock_act_name
        else:
            symbol_name = symbol_name
            symbol_code = symbol_code
        return symbol_name, symbol_code

    @lru_cache(maxsize=3)
    def get_stock_mapping(self, market_name: str):
        if market_name == "CN-Stock":
            stock_df = pro_cached.run(
                func_name="stock_basic",
                func_kwargs={
                    "exchange": "",
                    "fields": 'ts_code,symbol,name,area,industry,list_date,list_status,fullname'
                }
            )
            stock_name2code = {}
            stock_code2name = {}
            stock_info_columns = stock_df.columns.tolist()
            for stock in stock_df.values.tolist():
                stock_info = {v:k for k,v in zip(stock, stock_info_columns)}
                stock_name = stock_info['name']
                stock_name2code[stock_name] = stock_info['ts_code']
                if '-' in stock_name:
                    stock_name = stock_name.split('-')[0]
                stock_name2code[stock_name] = stock_info['ts_code']
                stock_code2name[stock_info['ts_code']] = stock_name

            # 读取曾用名, 包括ST
            name2code = self.get_total_namechange(market_name)
            for name, code in name2code.items():
                stock_name2code[name] = code

        return stock_name2code, stock_code2name

    def get_total_namechange(self, market_name: str):

        if market_name != "CN-Stock":
            return {}

        cache_path = Path(__file__).parent / 'cache' / 'market_manager' / 'namechange_data.json'
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Required cache file not found: {cache_path}. "
                "Please ensure the cache file exists."
            )
        
        with open(cache_path, 'r', encoding='utf-8') as f:
            name2code = json.load(f)
        
        return name2code

GLOBAL_MARKET_CONFIG = MarketManagerConfig.from_config_file()
GLOBAL_MARKET_MANAGER = MarketManager(GLOBAL_MARKET_CONFIG)

if __name__ == "__main__":
    print(GLOBAL_MARKET_MANAGER.get_target_symbol_context("2025-01-01 09:00:00"))
    print(GLOBAL_MARKET_MANAGER.get_total_namechange("CN-Stock"))
    pass

