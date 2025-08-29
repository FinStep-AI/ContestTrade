import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from matplotlib import rcParams
import sys
import os

# --- Imports from project ---
# A股相关功能可选导入；在美股模式下不要求安装tushare
try:
    from utils.tushare_utils import pro_cached
except Exception:
    pro_cached = None
from utils.yahoo_utils import get_yahoo_stock_price, get_yahoo_stock_info
from utils.finnhub_utils import finnhub_cached
from utils.date_utils import get_previous_trading_date

# --- Matplotlib Setup for Chinese Characters ---
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

#==============================================================================
# INTRADAY DATA FUNCTIONS (Restored with Placeholder)
#==============================================================================

def _get_intraday_data(stock_code, trade_date=None):
    try:
        year = trade_date[:4]
        code = stock_code.split('.')[0]
        market = stock_code.split('.')[1].upper()
        file_name = f"{market}.{code}.csv"
        file_path = f"/REPLACE_WITH_YOUR_LOCAL_DATA_PATH/stock_price/{year}/{file_name}"
        
        if not os.path.exists(file_path):
            return None, f"Intraday data file not found at placeholder path: {file_path}"
        
        df = pd.read_csv(file_path)
        date_str = datetime.strptime(trade_date, '%Y%m%d').strftime('%Y-%m-%d')
        df['日期'] = pd.to_datetime(df['日期'])
        df = df[df['日期'].dt.strftime('%Y-%m-%d') == date_str]
        
        if df.empty: return None, "No intraday data for the specified date."
        
        data_list = []
        for _, row in df.iterrows():
            data_list.append({'datetime_obj': row['日期'], 'last_price': row['收盘价'], 'trade_lots': row['成交量（手）'], 'preclose_price': None, 'total_trade_balance': row['成交额（元）']})
        
        if data_list:
            preclose = data_list[0]['last_price']
            for d in data_list: d['preclose_price'] = preclose
        
        return {'data': data_list}, None
    except Exception as e:
        return None, str(e)

def _generate_intraday_chart_base64(data, stock_code, stock_name, report_date=None):
    df = pd.DataFrame(data['data']).sort_values('datetime_obj').reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.15})
    preclose_price = df['preclose_price'].iloc[0]
    x_positions = list(range(len(df)))
    ax1.plot(x_positions, df['last_price'], color='#1f77b4', linewidth=1)
    ax1.axhline(y=preclose_price, color='#888888', linestyle='-', alpha=0.5)
    ax1.set_title(f'{stock_name}({stock_code}) 分时走势图 - {report_date}', fontsize=14, fontweight='bold')
    ax2.bar(x_positions, df['trade_lots'], color='#ff7f0e', alpha=0.7)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def _describe_intraday_data(intraday_data):
    prices = [item['last_price'] for item in intraday_data['data']]
    price_change = prices[-1] - intraday_data['data'][0]['preclose_price']
    return f"""分时数据摘要:
当前价格: {prices[-1]:.2f}元
前收盘价: {intraday_data['data'][0]['preclose_price']:.2f}元
涨跌额: {price_change:+.2f}元
涨跌幅: {(price_change / intraday_data['data'][0]['preclose_price'] * 100):+.2f}%
最高价: {max(prices):.2f}元
最低价: {min(prices):.2f}元"""

#==============================================================================
# K-LINE DATA FUNCTIONS (1:1 with original get_kline_description.py)
#==============================================================================

def _get_kline_data(stock_code, kline_num, end_date):
    df = pro_cached.run("daily", func_kwargs={'ts_code': stock_code, 'start_date': (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=kline_num * 2)).strftime('%Y%m%d'), 'end_date': end_date}, verbose=False)
    if df is None or df.empty: return None
    df = df.sort_values('trade_date', ascending=False).head(kline_num).sort_values('trade_date', ascending=True)
    return {'data': [{'trade_date': r['trade_date'], 'open_price': r['open'], 'high_price': r['high'], 'low_price': r['low'], 'close_price': r['close'], 'preclose_price': r['pre_close'], 'price_change': r['change'], 'price_change_rate': r['pct_chg'], 'volume': r['vol'], 'trade_amount': r['amount'], 'trade_lots': r['vol']} for _, r in df.iterrows()], 'stock_code': stock_code, 'kline_num': len(df)}

def _get_kline_data_us(symbol, kline_num, end_date):
    df = get_yahoo_stock_price(symbol=symbol, start_date=(datetime.strptime(end_date, '%Y%m%d') - timedelta(days=kline_num * 2)).strftime('%Y-%m-%d'), end_date=datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d'), verbose=False)
    if df is None or df.empty: return None
    df = df.sort_values('date', ascending=False).head(kline_num).sort_values('date', ascending=True).reset_index(drop=True)
    kline_data = []
    for i, r in df.iterrows():
        preclose = df.iloc[i-1]['close'] if i > 0 else r['close']
        kline_data.append({'trade_date': r['date'].strftime('%Y%m%d'), 'open_price': float(r['open']), 'high_price': float(r['high']), 'low_price': float(r['low']), 'close_price': float(r['close']), 'preclose_price': float(preclose), 'price_change': float(r['close'] - preclose), 'price_change_rate': (float(r['close'] - preclose) / preclose * 100) if preclose != 0 else 0, 'volume': int(r['volume']), 'trade_amount': float(r['volume'] * r['close']), 'trade_lots': int(r['volume'])})
    return {'data': kline_data, 'stock_code': symbol, 'kline_num': len(kline_data)}

def _generate_kline_chart_base64(data, stock_code, stock_name, report_date, currency_symbol="元", volume_unit="手"):
    if not data or 'data' not in data or not data['data']: return None
    df = pd.DataFrame(data['data'])
    df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.sort_values('date').reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.2})
    x_pos = np.arange(len(df))
    for i, r in df.iterrows():
        color = '#ff6b6b' if r['close_price'] >= r['open_price'] else '#51cf66'
        ax1.plot([i, i], [r['low_price'], r['high_price']], color=color, linewidth=1)
        rect = plt.Rectangle((i - 0.3, min(r['open_price'], r['close_price'])), 0.6, abs(r['close_price'] - r['open_price']), facecolor=color, edgecolor=color)
        ax1.add_patch(rect)
    for ma in [5, 10, 20, 60]:
        if len(df) >= ma: ax1.plot(x_pos, df['close_price'].rolling(window=ma).mean(), linewidth=1.5, label=f'MA{ma}')
    ax1.set_title(f'{stock_name}({stock_code}) K线图 - {report_date}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax2.bar(x_pos, df['volume'], color='#ff7f0e', alpha=0.7, width=0.8)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def _describe_kline_data(kline_data, currency_symbol="元", volume_unit="手", amount_unit="万"):
    if not kline_data or 'data' not in kline_data or not kline_data['data']: return "K线数据不可用"
    data_list = kline_data['data']
    recent_data = data_list[-7:]
    df_data = [{'日期': f"{d['trade_date'][:4]}-{d['trade_date'][4:6]}-{d['trade_date'][6:8]}", '开盘价': f"{d['open_price']:.2f}", '最高价': f"{d['high_price']:.2f}", '最低价': f"{d['low_price']:.2f}", '收盘价': f"{d['close_price']:.2f}", '涨跌额': f"{d['price_change']:+.2f}", '涨跌幅(%)': f"{d.get('price_change_rate', 0):+.2f}", f'成交量({volume_unit})': f"{d['trade_lots']:,.0f}", f'成交额({amount_unit})': f"{d['trade_amount']:,.0f}"} for d in recent_data]
    df = pd.DataFrame(df_data)
    closes = [item['close_price'] for item in data_list]
    ma_data = []
    for i, item in enumerate(recent_data):
        idx = len(data_list) - len(recent_data) + i
        ma_data.append({'MA5': f"{sum(closes[idx-4:idx+1])/5:.2f}" if idx >= 4 else '-', 'MA10': f"{sum(closes[idx-9:idx+1])/10:.2f}" if idx >= 9 else '-', 'MA20': f"{sum(closes[idx-19:idx+1])/20:.2f}" if idx >= 19 else '-'}) 
    ma_df = pd.DataFrame(ma_data)
    df = pd.concat([df, ma_df], axis=1)
    return f"""## 近{len(recent_data)}天K线数据详情

{df.to_markdown(index=False, tablefmt='pipe')}

### 数据说明
- 数据时间范围: {df.iloc[0]['日期']} 至 {df.iloc[-1]['日期']}
- MA5/MA10/MA20: 分别为5日、10日、20日移动平均线
- 成交量单位: {volume_unit}
- 成交额单位: {amount_unit}
- 价格单位: {currency_symbol}"""

#==============================================================================
# FINANCIAL DATA FUNCTIONS (1:1 with original get_financial_info.py)
#==============================================================================

def _get_financial_analysis(market, stock_code, stock_name, end_date):
    if market == "CN-Stock":
        df = pro_cached.run("fina_indicator", func_kwargs={'ts_code': stock_code, 'start_date': (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=365)).strftime('%Y%m%d'), 'end_date': end_date}, verbose=False)
        if df is None or df.empty: return {'success': False, 'summary': "财务数据暂不可用"}
        return {'success': True, 'summary': _generate_cn_financial_narrative(df.sort_values('end_date', ascending=False).iloc[0].to_dict(), stock_name)}
    elif market == "US-Stock":
        basic = finnhub_cached.run('company_basic_financials', {'symbol': stock_code, 'metric': 'all'}, verbose=False)
        if not basic: return {'success': False, 'summary': f"Failed to fetch financial data for {stock_code}"}
        reported = finnhub_cached.run('financials_reported', {'symbol': stock_code, 'freq': 'annual'}, verbose=False)
        return {'success': True, 'summary': _create_us_financial_narrative({'symbol': stock_code, 'basic_metrics': basic, 'financials_reported': reported}, stock_name)}
    return {'success': False, 'summary': f"Market type '{market}' is not supported."}

def _generate_cn_financial_narrative(data, stock_name):
    def f(k, p=False, d="数据缺失"): v = data.get(k); return d if v is None or pd.isna(v) else f"{float(v):.2f}%" if p else str(v)
    return f"""{stock_name}财务基本面分析：

盈利能力分析：
- 基本每股收益(EPS)：{f('eps')}元
- 净资产收益率(ROE)：{f('roe', p=True)}
- 销售毛利率：{f('grossprofit_margin', p=True)}
- 销售净利率：{f('netprofit_margin', p=True)}

成长性分析：
- 净利润同比增长率：{f('netprofit_yoy', p=True)}
- 营业收入同比增长率：{f('or_yoy', p=True)}
- 净资产同比增长率：{f('eqt_yoy', p=True)}

每股指标：
- 每股净资产：{f('bps')}元
- 每股营业总收入：{f('total_revenue_ps')}元
- 每股经营活动现金流量：{f('ocfps')}元

偿债能力分析：
- 资产负债率：{f('debt_to_assets', p=True)}
- 流动比率：{f('current_ratio')}
- 速动比率：{f('quick_ratio')}

报告期：{f('end_date')}，公告日期：{f('ann_date')}"""

def _create_us_financial_narrative(us_financial_result, stock_name):
    if not us_financial_result:
        return f"Financial data for {stock_name} is not available."
    
    symbol = us_financial_result['symbol']
    basic_metrics = us_financial_result.get('basic_metrics', {})
    financials_reported = us_financial_result.get('financials_reported', {})
    
    metrics = basic_metrics.get('metric', {})
    
    # Get latest financial report data
    latest_report = {}
    year = "N/A"
    if financials_reported and 'data' in financials_reported and financials_reported['data']:
        latest_data = financials_reported['data'][0]
        year = latest_data.get('year', 'N/A')
        report = latest_data.get('report', {})
        
        if 'ic' in report:  # Income Statement
            for item in report['ic']:
                concept = item.get('concept', '').replace('us-gaap_', '')
                latest_report[concept] = item.get('value', 0)
        if 'bs' in report:  # Balance Sheet
            for item in report['bs']:
                concept = item.get('concept', '').replace('us-gaap_', '')
                latest_report[concept] = item.get('value', 0)
    
    def safe_get_value(key, default="N/A"):
        value = metrics.get(key)
        if value is None or pd.isna(value):
            return default
        return value
    
    def format_percentage(value):
        if value is None or pd.isna(value):
            return "N/A"
        try:
            return f"{float(value):.2f}%"
        except (ValueError, TypeError):
            return str(value)
    
    def format_number(value, decimal_places=2):
        if value is None or pd.isna(value):
            return "N/A"
        try:
            return f"{float(value):.{decimal_places}f}"
        except (ValueError, TypeError):
            return str(value)
    
    def format_large_number(value):
        if value is None or pd.isna(value):
            return "N/A"
        try:
            value_float = float(value)
            if value_float >= 1000000000:  # Billions
                return f"${value_float/1000000000:.2f}B"
            elif value_float >= 1000000:  # Millions
                return f"${value_float/1000000:.2f}M"
            else:
                return f"${value_float:,.2f}"
        except (ValueError, TypeError):
            return str(value)
    
    # Build English narrative
    narrative = f"""
{stock_name} ({symbol}) Financial Analysis (Latest Report Year: {year}):

PROFITABILITY ANALYSIS:
- Earnings Per Share (Diluted): {format_number(latest_report.get('EarningsPerShareDiluted', metrics.get('epsInclExtraItemsTTM')))}
- Return on Equity (ROE): {format_percentage(safe_get_value('roeTTM'))}
- Return on Assets (ROA): {format_percentage(safe_get_value('roaTTM'))}
- Gross Margin: {format_percentage(safe_get_value('grossMarginTTM'))}
- Net Margin: {format_percentage(safe_get_value('netProfitMarginTTM'))}

GROWTH ANALYSIS:
- Revenue Growth (TTM YoY): {format_percentage(safe_get_value('revenueGrowthTTMYoy'))}
- EPS Growth (TTM YoY): {format_percentage(safe_get_value('epsGrowthTTMYoy'))}

PER SHARE METRICS:
- Book Value Per Share: {format_number(safe_get_value('bookValuePerShareAnnual'))}
- Revenue Per Share: {format_number(safe_get_value('salesPerShareTTM'))}
- Cash Per Share: {format_number(safe_get_value('cashPerSharePerShareTTM'))}

DEBT & LIQUIDITY ANALYSIS:
- Current Ratio: {format_number(safe_get_value('currentRatioAnnual'))}
- Quick Ratio: {format_number(safe_get_value('quickRatioAnnual'))}
- Debt to Equity: {format_number(safe_get_value('totalDebtToEquityAnnual'))}

VALUATION METRICS:
- P/E Ratio (TTM): {format_number(safe_get_value('peInclExtraTTM'))}
- P/B Ratio: {format_number(safe_get_value('pbAnnual'))}
- Price to Sales: {format_number(safe_get_value('psAnnual'))}

KEY FINANCIAL FIGURES:
- Total Revenue: {format_large_number(latest_report.get('RevenueFromContractWithCustomerExcludingAssessedTax'))}
- Net Income: {format_large_number(latest_report.get('NetIncomeLoss'))}
- Total Assets: {format_large_number(latest_report.get('Assets'))}
- Total Equity: {format_large_number(latest_report.get('StockholdersEquity'))}"""

    return narrative.strip()

#==============================================================================
# OTHER CN-DATA FUNCTIONS (1:1 with originals, BUG FIXED)
#==============================================================================

def _get_sector_analysis_dc(end_date, stock_code):
    df = pro_cached.run("moneyflow_ind_dc", func_kwargs={'trade_date': end_date}, verbose=False)
    member_df = pro_cached.run("dc_member", func_kwargs={'con_code': stock_code}, verbose=False)
    if df is None or member_df is None or member_df.empty: return {'success': False, 'message': '未找到股票的板块信息'}
    sector_codes = member_df['ts_code'].unique().tolist()
    sector_info = df[df['ts_code'].isin(sector_codes)]
    if sector_info.empty: return {'success': False, 'message': '未找到板块的资金流向信息'}
    return {'success': True, 'data': sector_info, 'stock_code': stock_code}

def _create_sector_dc_prompt(sector_result):
    data = sector_result['data']
    stock_code = sector_result['stock_code']
    prompt = f"""多板块资金流向分析 (日期: {data.iloc[0]['trade_date']}):

股票代码: {stock_code}
所属板块数量: {len(data)}个

板块详细信息（按资金净流入排序）:
"""
    for i, (_, r) in enumerate(data.iterrows(), 1):
        prompt += f"\n{i}. 板块名称: {r['name']}\n   板块排名: 第{r['rank']}位\n   资金净流入: {r['net_amount']/1e8:.2f}亿元\n   板块涨跌幅: {r['pct_change']:.2f}%\n   板块收盘价: {r['close']:.2f}\n   板块代码: {r['ts_code']}"
    prompt += "\n\n请基于以上该股票所属的多个板块资金流向数据，分析各板块的资金偏好、市场表现差异，并评估该股票在不同板块中的定位."
    return prompt.strip()

def _get_stock_recent_3days_moneyflow_dc(market, stock_code, end_date):
    if market != "CN-Stock": return {'success': False, 'message': f"Market type '{market}' not supported."}
    df = pro_cached.run("moneyflow_dc", func_kwargs={'ts_code': stock_code, 'start_date': (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=5)).strftime('%Y%m%d'), 'end_date': end_date}, verbose=False)
    if df is None or df.empty: return {'success': False, 'message': '没有找到个股资金流向数据'}
    return {'success': True, 'data': df.sort_values('trade_date', ascending=False).head(3)}

def _create_recent_3days_dc_narrative(symbol, moneyflow_data):
    narrative = f"{symbol}近三日资金流向分析：\n\n"
    total_net_inflow = 0
    for i, (_, r) in enumerate(moneyflow_data['data'].iterrows(), 1):
        total_net_inflow += r['net_amount']
        narrative += f"第{i}日({r['trade_date']}): 总净流入{r['net_amount']/1e4:.2f}亿元，主力净流入{r['buy_lg_amount']/1e4:.2f}亿元，散户净流入{r['buy_elg_amount']/1e4:.2f}亿元，涨跌幅:{r['pct_change']:.2f}%\n"
    narrative += f"\n三日累计{'净流入' if total_net_inflow > 0 else '净流出'}{abs(total_net_inflow/1e4):.2f}亿元"
    return narrative.strip()

def _get_technical_analysis(market, stock_code, stock_name, end_date, days=3):
    if market != "CN-Stock": return {'success': False, 'message': f"Market type '{market}' not supported."}
    df = pro_cached.run("stk_factor_pro", func_kwargs={'ts_code': stock_code, 'start_date': (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=15)).strftime('%Y%m%d'), 'end_date': end_date}, verbose=False)
    if df is None or df.empty: return {'success': False, 'message': '没有找到技术面因子数据'}
    return {'success': True, 'factor_result': df.sort_values('trade_date', ascending=False).iloc[0].to_dict()}

def _generate_technical_narrative(factor_result, stock_name):
    data = factor_result
    def f(k, d=0): v = data.get(k); return d if v is None or pd.isna(v) else v
    rsi, macd, k, d_k, j = f('rsi_bfq_12'), f('macd_bfq'), f('kdj_k_bfq'), f('kdj_d_bfq'), f('kdj_bfq')
    boll_u, boll_m, boll_l = f('boll_upper_bfq'), f('boll_mid_bfq'), f('boll_lower_bfq')
    ma5, ma10, ma20 = f('ma_bfq_5'), f('ma_bfq_10'), f('ma_bfq_20')
    vol_ratio = f('vol') / f('vol_ratio', 1) if f('vol_ratio') else 1
    return f"""{stock_name}技术面因子分析：

基础行情：
- 当前价格: {f('close'):.2f}元
- 涨跌幅: {f('pct_chg'):.2f}%
- 换手率: {f('turnover_rate'):.2f}%

动量指标：
- RSI(12): {rsi:.1f} {'(超买区间)' if rsi > 70 else '(超卖区间)' if rsi < 30 else '(正常区间)'}
- MACD: {macd:.3f} {'(金叉信号)' if macd > 0 else '(死叉信号)'}

KDJ指标：
- K值: {k:.1f}
- D值: {d_k:.1f}
- J值: {j:.1f}
- 状态: {'金叉' if k > d_k else '死叉'}

布林带分析：
- 上轨: {boll_u:.2f}
- 中轨: {boll_m:.2f}
- 下轨: {boll_l:.2f}
- 当前价格: {f('close'):.2f}
- 位置: {'上轨附近' if f('close') > boll_u * 0.98 else '下轨附近' if f('close') < boll_l * 1.02 else '中轨附近'}

均线系统：
- MA5: {ma5:.2f}
- MA10: {ma10:.2f}
- MA20: {ma20:.2f}
- 均线排列: {'多头排列' if ma5 > ma10 > ma20 else '空头排列' if ma5 < ma10 < ma20 else '震荡排列'}

成交量指标：
- 量比: {vol_ratio:.1f} {'(放量)' if vol_ratio > 1.5 else '(缩量)' if vol_ratio < 0.8 else '(正常)'}

估值指标：
- 市盈率(PE): {f('pe'):.2f}
- 市净率(PB): {f('pb'):.2f}"""

#==============================================================================
# MAIN PUBLIC FUNCTION
#==============================================================================

def get_all_stock_data(market, symbol, stock_name, trigger_time):
    results = {
        "intraday_description": "分时数据暂不可用", "intraday_chart_base64": None,
        "kline_description": "K线数据暂不可用", "kline_chart_base64": None,
        "financial_summary": "财务数据暂不可用", "sector_analysis": "板块资金流向数据暂不可用",
        "stock_moneyflow_analysis": "个股资金流向数据暂不可用", "technical_analysis": "技术面因子数据暂不可用",
    }
    previous_trading_date = get_previous_trading_date(trigger_time)

    if market == "CN-Stock":
        intraday_data, err = _get_intraday_data(symbol, previous_trading_date)
        if intraday_data and intraday_data.get('data'):
            results["intraday_chart_base64"] = _generate_intraday_chart_base64(intraday_data, symbol, stock_name, previous_trading_date)
            results["intraday_description"] = _describe_intraday_data(intraday_data)
            print(f"✅  分时数据获取成功，图表已生成 (使用占位符路径逻辑)")
        else:
            print(f"❌ 分时数据获取失败")

        kline_data = _get_kline_data(symbol, 90, previous_trading_date)
        if kline_data and kline_data.get('data'):
            results["kline_chart_base64"] = _generate_kline_chart_base64(kline_data, symbol, stock_name, previous_trading_date)
            results["kline_description"] = _describe_kline_data(kline_data)
            print(f"✅  K线数据获取成功, 共{len(kline_data['data'])}条记录，图表已生成")
        else: print(f"❌  K线数据获取失败")
        
        fin_res = _get_financial_analysis(market, symbol, stock_name, previous_trading_date)
        if fin_res['success']:
            results["financial_summary"] = fin_res['summary']
            print(f"✅  财务数据获取成功")
        else: print(f"❌  财务数据获取失败: {fin_res.get('summary')}")

        sec_res = _get_sector_analysis_dc(previous_trading_date, symbol)
        if sec_res['success']:
            results["sector_analysis"] = _create_sector_dc_prompt(sec_res)
            print(f"✅  板块资金流向获取成功")
        else: print(f"❌  板块资金流向获取失败: {sec_res['message']}")

        mon_res = _get_stock_recent_3days_moneyflow_dc(market, symbol, previous_trading_date)
        if mon_res['success']:
            results["stock_moneyflow_analysis"] = _create_recent_3days_dc_narrative(symbol, mon_res)
            print(f"✅  个股资金流向获取成功")
        else: print(f"❌  个股资金流向获取失败: {mon_res['message']}")

        tech_res = _get_technical_analysis(market, symbol, stock_name, previous_trading_date)
        if tech_res['success']:
            results["technical_analysis"] = _generate_technical_narrative(tech_res['factor_result'], stock_name)
            print(f"✅  技术面因子数据获取成功")
        else: print(f"❌  技术面因子数据获取失败: {tech_res['message']}")

    elif market == "US-Stock":
        results["intraday_description"] = f"美股({symbol})暂不支持分时数据"
        kline_data = _get_kline_data_us(symbol, 90, previous_trading_date)
        if kline_data and kline_data.get('data'):
            results["kline_chart_base64"] = _generate_kline_chart_base64(kline_data, symbol, stock_name, previous_trading_date, "$", "share")
            results["kline_description"] = _describe_kline_data(kline_data, "$", "share", "k$")
            print(f"✅  K线数据获取成功, 共{len(kline_data['data'])}条记录，图表已生成")
        else: print(f"❌  K线数据获取失败")

        fin_res = _get_financial_analysis(market, symbol, stock_name, previous_trading_date)
        if fin_res['success']:
            results["financial_summary"] = fin_res['summary']
            print(f"✅  财务数据获取成功")
        else: print(f"❌  财务数据获取失败: {fin_res.get('summary')}")

    return results
