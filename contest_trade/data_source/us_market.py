"""
US Market data source
生成美股宏观市场概览（大盘ETF行情 + 行业ETF简要 + 要闻摘要）
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

from data_source.data_source_base import DataSourceBase
from utils.date_utils import get_previous_trading_date
from utils.yahoo_utils import get_yahoo_stock_price, get_yahoo_financial_news


def _fmt_pct(x: float) -> str:
    try:
        return f"{x*100:+.2f}%"
    except Exception:
        return "N/A"


def _fetch_close_and_change(symbol: str, trade_date_yyyymmdd: str):
    """获取某美股标的在目标交易日及前一日的收盘价与涨跌幅"""
    try:
        # 拉取近7日，避免遇到非交易日
        end_dt = datetime.strptime(trade_date_yyyymmdd, "%Y%m%d")
        start_dt = end_dt - timedelta(days=7)
        df = get_yahoo_stock_price(symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), verbose=False)
        if df is None or df.empty:
            return None
        # 取目标交易日记录
        df = df.sort_values('date')
        df['date_str'] = df['date'].dt.strftime('%Y%m%d')
        cur_row = df[df['date_str'] == trade_date_yyyymmdd].tail(1)
        if cur_row.empty:
            return None
        # 取前一条可用记录
        idx = cur_row.index[0]
        prev_idx = df.index.get_loc(idx) - 1
        if prev_idx < 0:
            prev_close = float(cur_row.iloc[0]['close'])
        else:
            prev_close = float(df.iloc[prev_idx]['close'])
        cur_close = float(cur_row.iloc[0]['close'])
        change = (cur_close - prev_close) / prev_close if prev_close else 0.0
        return {
            'symbol': symbol,
            'close': cur_close,
            'prev_close': prev_close,
            'change': change,
        }
    except Exception:
        return None


class USMarket(DataSourceBase):
    def __init__(self):
        super().__init__("us_market")

    def _build_overview(self, trade_date_yyyymmdd: str) -> str:
        # 大盘ETF
        major = ['SPY', 'QQQ', 'DIA']
        # 行业ETF（示例）
        sectors = ['XLK', 'XLF', 'XLE']

        lines = []
        lines.append(f"美股市场日度概览（{trade_date_yyyymmdd[:4]}-{trade_date_yyyymmdd[4:6]}-{trade_date_yyyymmdd[6:]}）")
        lines.append("\n[大盘ETF]")
        for s in major:
            r = _fetch_close_and_change(s, trade_date_yyyymmdd)
            if r:
                lines.append(f"- {s}: 收 {r['close']:.2f} ({_fmt_pct(r['change'])})")
        lines.append("\n[行业ETF]")
        for s in sectors:
            r = _fetch_close_and_change(s, trade_date_yyyymmdd)
            if r:
                lines.append(f"- {s}: 收 {r['close']:.2f} ({_fmt_pct(r['change'])})")

        # 要闻
        try:
            news_df = get_yahoo_financial_news(limit=8, verbose=False)
            if news_df is not None and not news_df.empty:
                news_df = news_df.head(5)
                lines.append("\n[当日要闻]")
                for _, row in news_df.iterrows():
                    headline = str(row.get('title', ''))[:120]
                    site = str(row.get('site', '') or row.get('source', ''))
                    lines.append(f"- {headline} ({site})")
        except Exception:
            pass

        return "\n".join(lines).strip()

    def get_data(self, trigger_time: str) -> pd.DataFrame:
        try:
            cached = self.get_data_cached(trigger_time)
            if cached is not None:
                return cached

            trade_date = get_previous_trading_date(trigger_time)  # YYYYMMDD
            content = self._build_overview(trade_date)
            data = [{
                'title': f"{trade_date}: US Market Overview",
                'content': content,
                'pub_time': trigger_time,
                'url': None,
            }]
            df = pd.DataFrame(data)
            self.save_data_cached(trigger_time, df)
            return df
        except Exception:
            return pd.DataFrame()


if __name__ == "__main__":
    ds = USMarket()
    test_time = "2025-08-12 09:00:00"
    print(ds.get_data(test_time))


