"""
tushare 的工具函数

1. 获取交易日列表
2. 获取股票日线数据
"""
import os
import json
import pandas as pd
import tushare as ts
from pathlib import Path
from datetime import datetime
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta
import hashlib
import pickle
from config.config import cfg

DEFAULT_TUSHARE_CACHE_DIR = Path(__file__).parent / "tushare_cache"

class CachedTusharePro:
    def __init__(self, cache_dir=None):
        if not cache_dir:
            self.cache_dir = DEFAULT_TUSHARE_CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        token = cfg.tushare_key
        ts.set_token(token)
        self.pro = ts.pro_api(token)

    def run(self, func_name: str, func_kwargs: dict, verbose: bool = False):
        func_kwargs_str = json.dumps(func_kwargs)
        return self.run_with_cache(func_name, func_kwargs_str, verbose)
    
    def run_with_cache(self, func_name: str, func_kwargs: str, verbose: bool = False):
        func_kwargs = json.loads(func_kwargs)
        args_hash = hashlib.md5(str(func_kwargs).encode()).hexdigest()
        func_cache_dir = self.cache_dir / func_name
        if not func_cache_dir.exists():
            func_cache_dir.mkdir(parents=True, exist_ok=True)
        func_cache_file = func_cache_dir / f"{args_hash}.pkl"
        if func_cache_file.exists():
            if verbose:
                print(f"load result from {func_cache_file}")
            with open(func_cache_file, "rb") as f:
                return pickle.load(f)
        else:
            if verbose:
                print(f"cache miss for {func_name} with args: {func_kwargs}")
            result = getattr(self.pro, func_name)(**func_kwargs)
            if verbose:
                print(f"save result to {func_cache_file}")
            with open(func_cache_file, "wb") as f:
                pickle.dump(result, f)
            return result

pro_cached = CachedTusharePro()

def fix_stock_code(stock_name, stock_code, verbose=False):
    stock_code = stock_code if not stock_code.startswith('s') else stock_code[2:] + '.' + stock_code[:2].upper()
    stock_code = stock_code.strip('.')
    if stock_code.startswith("S"):
        stock_code = stock_code[2:] + '.' + stock_code[:2].upper()
    stock_name2code, stock_code2name = get_stock_mapping()
    stock_act_code = stock_name2code.get(stock_name, "xxxxx")
    if stock_act_code != stock_code:
        if stock_act_code != "xxxxx":
            if verbose:
                print(f"set stock_code: {stock_code} to {stock_act_code}")
            stock_code = stock_act_code
    stock_act_name = stock_code2name.get(stock_code, "xxxxx")
    if stock_act_name in stock_name and stock_act_name != stock_name:
        if verbose:
            print(f"set stock_name: {stock_name} to {stock_act_name}")
        if stock_act_name != "xxxxx":
            stock_name = stock_act_name
    return stock_name, stock_code

@lru_cache(maxsize=1)
def get_trade_date(cache_dir=None, verbose=False):
    if cache_dir is None:
        cache_dir = DEFAULT_TUSHARE_CACHE_DIR

    if cache_dir and not Path(cache_dir).exists():
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if cache_dir:
        cache_file = Path(cache_dir) / f"trade_date.csv"
    else:
        cache_file = None

    if cache_file and cache_file.exists():
        trade_date = pd.read_csv(cache_file)
        if verbose:
            print(f"load trade_date from {cache_file} success")
    else:
        if verbose:
            print("load trade_date from tushare")
        trade_date = pro.trade_cal(exchange="SSE")
        trade_date.to_csv(cache_file, index=False)
        if verbose:
            print("load trade_date from tushare success")

    trade_date_list = [str(d) for d in trade_date[trade_date["is_open"] == 1]["cal_date"].values.tolist()]
    trade_date_list.sort()
    return trade_date_list


@lru_cache(maxsize=1)
def get_stock_basic(update_date=None, cache_dir=None, detail=False, verbose=None):
    """
    读取股票基本信息
    """
    if not update_date:
        update_date = datetime.now().strftime("%Y%m%d")

    if cache_dir is None:
        cache_dir = DEFAULT_TUSHARE_CACHE_DIR

    if cache_dir and not Path(cache_dir).exists():
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if cache_dir:
        cache_file = Path(cache_dir) / f"stock_infos_{update_date}.csv"
    else:
        cache_file = None

    if cache_file and cache_file.exists():
        stock_infos = pd.read_csv(cache_file)
        if verbose:
            print(f"load stock_infos from {cache_file} success")
    else:
        if verbose:
            print("load stock_infos from tushare")
        stock_infos = pro.stock_basic(exchange="SSE,SZSE,BJSE", fields='ts_code,symbol,name,area,industry,list_date,list_status,fullname')
        stock_infos.to_csv(cache_file, index=False)
        if verbose:
            print("load stock_infos from tushare success")

    stock_list = stock_infos[stock_infos['list_status'] == 'L']['ts_code'].values.tolist()
    if detail:
        return stock_infos
    return stock_list


@lru_cache(maxsize=10000)
def get_daily_klines(stock_code, start_date, end_date, cache_dir=None, verbose=False):
    if cache_dir is None:
        cache_dir = DEFAULT_TUSHARE_CACHE_DIR

    if cache_dir and not Path(cache_dir).exists():
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if cache_dir:
        cache_file = Path(cache_dir) / f"{stock_code}.csv"
    else:
        cache_file = None
    
    load_ok = False
    if cache_file and cache_file.exists():
        df = pd.read_csv(cache_file)
        df['trade_date'] = df['trade_date'].astype(str)
        # check if the cache date is valid
        df = df.sort_values(by="trade_date", ascending=False)
        trade_date = df["trade_date"].values.tolist()
        cur_date = datetime.now().strftime("%Y%m%d")
        inner_trade_date = [d for d in get_trade_date(cache_dir=cache_dir, verbose=verbose) if d >= start_date and d <= end_date]
        inner_trade_date = [d for d in inner_trade_date if d < cur_date]
        inner_trade_date.sort()
        # check if the cache is valid
        if str(trade_date[-1]) <= inner_trade_date[0] and str(trade_date[0]) >= inner_trade_date[-1]:
            load_ok = True
            if verbose:
                print(f"load cache file {cache_file} success")
        else:
            if verbose:
                print(f"trade_date is not valid, {trade_date[-1]} <= {inner_trade_date[0]} and {trade_date[0]} >= {inner_trade_date[-1]}")

    if not load_ok:
        df = pro.daily(ts_code=stock_code)
        df['trade_date'] = df['trade_date'].astype(str)
        if cache_file and df is not None:
            df.to_csv(cache_file, index=False)
            load_ok = True

    if not load_ok:
        return None

    df = df[df['trade_date'] >= start_date]
    df = df[df['trade_date'] <= end_date]
    return df


@lru_cache(maxsize=10000)
def get_daily_limit_price(date, stock_code=None, cache_dir=None, verbose=False):
    """
    获取股票在指定日期的涨停价格
    """
    if cache_dir is None:
        cache_dir = DEFAULT_TUSHARE_CACHE_DIR

    if cache_dir and not Path(cache_dir).exists():
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if cache_dir:
        cache_file = Path(cache_dir) / f"limit_price_{date}.csv"
    else:
        cache_file = None

    if cache_file and cache_file.exists():
        limit_df = pd.read_csv(cache_file)
        if verbose:
            print(f"load limit_price from {cache_file} success")
    else:
        limit_df = pro.stk_limit(trade_date=date)
        limit_df.to_csv(cache_file, index=False)
        if verbose:
            print(f"save limit_price to {cache_file} success")
    
    if stock_code is not None:
        limit_df = limit_df[limit_df['ts_code'] == stock_code]
    return limit_df


@lru_cache(maxsize=1)
def get_stock_mapping():
    stock_df = get_stock_basic(detail=True)
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
    name2code = get_total_namechange()
    for name, code in name2code.items():
        stock_name2code[name] = code

    return stock_name2code, stock_code2name



@lru_cache(maxsize=100000)
def calc_pct(stock_code, buy_date, hold_days, buy_stg='open', sell_stg='open', buylimit=False, verbose=False):
    """
    计算股票在指定日期买入，持有指定天数后卖出的收益率
    """
    # 确定当天能不能买
    limit_df = get_daily_limit_price(buy_date, stock_code=stock_code, verbose=verbose)
    if limit_df is None:
        if verbose:
            print(f"can't get limit_price for stock {stock_code} at {buy_date}")
        return None
    if stock_code not in limit_df['ts_code'].values.tolist():
        if verbose:
            print(f"stock {stock_code} is not available at {buy_date}")
        return None

    date_after = [d for d in get_trade_date(verbose=verbose) if d >= buy_date]
    if buy_date not in date_after:
        print(f"can't buy stock {stock_code} at {buy_date}")
        return None
    sell_date = date_after[hold_days]

    if verbose:
        print(f"get a trade to buy {stock_code} at {buy_date} and sell at {sell_date}")
    df = get_daily_klines(stock_code, buy_date, sell_date)

    df_date = df['trade_date'].values.tolist()
    if buy_date not in df_date:
        if verbose:
            print(f"can't buy stock {stock_code} at {buy_date}")
        return None
    if sell_date not in df_date:
        if verbose:
            print(f"can't sell stock {stock_code} at {sell_date}")
        return None

    buy_price = df[df['trade_date'] == buy_date][buy_stg].values[0]
    sell_price = df[df['trade_date'] == sell_date][sell_stg].values[0]

    # 涨停不能买入
    if not buylimit:
        up_limit = limit_df['up_limit'].values[0]
        down_limit = limit_df['down_limit'].values[0]
        if buy_price >= up_limit:
            if verbose:
                print(f"stock {stock_code} at {buy_date} is up_limit, can't buy")
            return None
    
    # 暂时不考虑跌停


    if verbose:
        print(f"buy_price: {buy_price}, sell_price: {sell_price}")
    pct = (sell_price - buy_price) / buy_price
    pct = round(pct, 3)
    return pct


@lru_cache(maxsize=1)
def get_total_namechange():
    name2code = {}
    stock_df = get_stock_basic(detail=True)
    stock_list = stock_df['ts_code'].values.tolist()
    for code in stock_list:
        df = get_namechange(code)
        code_his_names = df['name'].values.tolist()
        for name in code_his_names:
            name2code[name] = code
    return name2code


@lru_cache(maxsize=10000)
def get_namechange(stock_code, cache_dir=None):
    if cache_dir is None:
        cache_dir = DEFAULT_TUSHARE_CACHE_DIR

    name_change_cache_dir = Path(cache_dir) / "namechange"
    if not name_change_cache_dir.exists():
        name_change_cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = name_change_cache_dir / f"{stock_code}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file)
    else:
        print(f"no cache file {cache_file}, get namechange from tushare")
        df = pro.namechange(ts_code=stock_code, fields='ts_code,name,start_date,end_date,change_reason')
        df.to_csv(cache_file, index=False)
    return df


@lru_cache(maxsize=10000)
def get_major_news(start_datetime, end_datetime, src='新浪财经', cache_dir=None, verbose=False):
    """
    获取新浪财经新闻数据
    
    Args:
        start_date: 开始日期，格式如 "20250101"
        end_date: 结束日期，格式如 "20250101"
        cache_dir: 缓存目录
        verbose: 是否输出详细信息
        src: 新闻来源，新闻来源（新华网、凤凰财经、同花顺、新浪财经、华尔街见闻、中证网、财新网、第一财经、财联社）
    """
    if cache_dir is None:
        cache_dir = DEFAULT_TUSHARE_CACHE_DIR

    dirname = src.strip() + '_news'
    news_cache_dir = Path(cache_dir) / dirname
    if not news_cache_dir.exists():
        news_cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = news_cache_dir / f"{src}_news_{start_datetime}_{end_datetime}.csv"
    
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        if len(df) > 10:
            rerun = False
            if verbose:
                print(f"load {src} news from {cache_file} success. with length {len(df)}")
        else:
            rerun = True
    else:
        rerun = True

    if rerun:
        if verbose:
            print(f"load {src} news from tushare for {start_datetime} to {end_datetime}")
        try:
            # convert from YYYYMMDD to YYYY-MM-DD HH:MM:SS
            df = pro.major_news(src=src, start_date=start_datetime, end_date=end_datetime, fields='title,content,pub_time,src_name')
            if df is not None and not df.empty:
                df.to_csv(cache_file, index=False)
                if verbose:
                    print(f"save {src} news to {cache_file} success")
            else:
                if verbose:
                    print(f"no {src} news data for {start_datetime} to {end_datetime}")
                return pd.DataFrame()
        except Exception as e:
            if verbose:
                print(f"error getting {src} news: {e}")
            return pd.DataFrame()
    return df

tushare_cached = CachedTusharePro()

if __name__ == "__main__":
    pass
