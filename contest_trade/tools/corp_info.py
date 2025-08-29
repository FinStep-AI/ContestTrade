"""
Crop Info Tools
"""
from pathlib import Path
import sys
import asyncio
import pandas as pd
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import asyncio
from langchain_core.tools import tool
from pydantic import BaseModel, Field
# A股相关功能可选导入
try:
    from utils.tushare_utils import tushare_cached
except Exception:
    tushare_cached = None
from tools.tool_prompts import FINANCIAL_TOOL_SELECT_PROMPT
from tools.tool_utils import ToolManager, ToolManagerConfig
from utils.finnhub_utils import finnhub_cached
from tools.tool_utils import smart_tool

class CompanyFinancialInput(BaseModel):
    market: str = Field(description="The market of the company.")
    symbol: str = Field(description="The symbol of the company.")
    task: str = Field(description="The query of the financial data.")
    trigger_time: str = Field(description="The trigger time of the financial data. Format: YYYY-MM-DD HH:MM:SS.")
@smart_tool(
    description="Get the financial information of a company",
    args_schema=CompanyFinancialInput,
    max_output_len=4000,
    timeout_seconds=30.0
)
async def company_financial_info(market: str, symbol: str, task: str, trigger_time: str=None) -> str:
    tools_config = ToolManagerConfig(tool_paths=[
        "tools.corp_info.company_income",
        "tools.corp_info.company_balance_sheet",
        "tools.corp_info.company_cash_flow",
        "tools.corp_info.company_forecast",
        "tools.corp_info.company_express",
        "tools.corp_info.company_dividend",
    ])
    inner_tool_manager = ToolManager(tools_config)
    tool_selection_prompt = FINANCIAL_TOOL_SELECT_PROMPT.format(
        date=trigger_time,
        task=task,
        market=market,
        symbol=symbol,
        tool_info=inner_tool_manager.build_toolcall_context()
    )
    tool_call = await inner_tool_manager.select_tool_by_llm(
        prompt=tool_selection_prompt,
    )
    if "error" in tool_call:
        return {"error": tool_call["error_msg"]}

    tool_result = await inner_tool_manager.call_tool(
        tool_call['tool_name'],
        tool_call['properties'],
        trigger_time
    )
    return tool_result


class CompanyIncomeInput(BaseModel):
    market: str = Field(description="The market of the company.")
    symbol: str = Field(description="The symbol of the company.")
    period: str = Field(description="The period of the financial data. Format: YYYYMMDD. \
                        Reporting period (the last day of each quarter, for example, \
                        20171231 for annual report, 20170630 for semi-annual report, 20170930 \
                        for third quarter report).")
    trigger_time: str = Field(description="The trigger time of the financial data. Format: YYYY-MM-DD HH:MM:SS.")
@tool(
    description="Get the income information of a company",
    args_schema=CompanyIncomeInput
)
async def company_income(market: str, symbol: str, period: str, trigger_time: str=None) -> str:
    if market == "CN-Stock":
        trigger_date = trigger_time.split(" ")[0].replace("-", "")
        if period < trigger_date:
            func_args = {
                "ts_code": symbol,
                "period": period
            }
            df = tushare_cached.run("income", func_kwargs=func_args)
            if df.empty:
                return {"error": f"No data found for income."}
            return df.to_markdown()
        else:
            return {"error": "The period is not in the trigger time."}
    elif market == "US-Stock":
        period_date = period[:4] + '-' + period[4:6] + '-' + period[6:8]
        datas = finnhub_cached.run('financials', {
            'symbol': symbol,
            'statement': 'ic',
            'freq': "quarterly"
        })['financials']
        data = [d for d in datas if d['period'] == period_date][0]
        return data
    else:
        return {"error": "Not supported yet."}

class CompanyBalanceSheetInput(BaseModel):
    market: str = Field(description="The market of the company.")
    symbol: str = Field(description="The symbol of the company.")
    period: str = Field(description="The period of the financial data. Format: YYYYMMDD. \
                        Reporting period (the last day of each quarter, for example, \
                        20171231 for annual report, 20170630 for semi-annual report, 20170930 \
                        for third quarter report).")
    trigger_time: str = Field(description="The trigger time of the financial data. Format: YYYY-MM-DD HH:MM:SS.")
@tool(
    description="Get the balance sheet information of a company",
    args_schema=CompanyBalanceSheetInput
)
async def company_balance_sheet(market: str, symbol: str, period: str, trigger_time: str=None) -> str:
    if market == "CN-Stock":
        trigger_date = trigger_time.split(" ")[0].replace("-", "")
        if period < trigger_date:
            func_args = {
                "ts_code": symbol,
                "period": period
            }
            df = tushare_cached.run("balancesheet", func_kwargs=func_args)
            if df.empty:
                return {"error": f"No data found for balance sheet."}
            return df.to_markdown()
        else:
            return {"error": "The period is not in the trigger time."}
    elif market == "US-Stock":
        period_date = period[:4] + '-' + period[4:6] + '-' + period[6:8]
        datas = finnhub_cached.run('financials', {
            'symbol': symbol,
            'statement': 'bs',
            'freq': "quarterly"
        })['financials']
        data = [d for d in datas if d['period'] == period_date][0]
        return data
    else:
        return {"error": "Not supported yet."}

class CompanyCashFlowInput(BaseModel):
    market: str = Field(description="The market of the company.")
    symbol: str = Field(description="The symbol of the company.")
    period: str = Field(description="The period of the financial data. Format: YYYYMMDD. \
                        Reporting period (the last day of each quarter, for example, \
                        20171231 for annual report, 20170630 for semi-annual report, 20170930 \
                        for third quarter report).")
    trigger_time: str = Field(description="The trigger time of the financial data. Format: YYYY-MM-DD HH:MM:SS.")
@tool(
    description="Get the cash flow information of a company",
    args_schema=CompanyCashFlowInput
)
async def company_cash_flow(market: str, symbol: str, period: str, trigger_time: str=None) -> str:
    if market == "CN-Stock":
        trigger_date = trigger_time.split(" ")[0].replace("-", "")
        if period < trigger_date:
            func_args = {
                "ts_code": symbol,
                "period": period
            }
            df = tushare_cached.run("cashflow", func_kwargs=func_args)
            if df.empty:
                return {"error": f"No data found for cash flow."}
            return df.to_markdown()
        else:
            return {"error": "The period is not in the trigger time."}
    elif market == "US-Stock":
        period_date = period[:4] + '-' + period[4:6] + '-' + period[6:8]
        datas = finnhub_cached.run('financials', {
            'symbol': symbol,
            'statement': 'cf',
            'freq': "quarterly"
        })['financials']
        data = [d for d in datas if d['period'] == period_date][0]
        return data
    else:
        return {"error": "Not supported yet."}


class CompanyForecastInput(BaseModel):
    market: str = Field(description="The market of the company.")
    symbol: str = Field(description="The symbol of the company.")
    period: str = Field(description="The period of the financial data. Format: YYYYMMDD. \
                        Reporting period (the last day of each quarter, for example, \
                        20171231 for annual report, 20170630 for semi-annual report, 20170930 \
                        for third quarter report).")
    trigger_time: str = Field(description="The trigger time of the financial data. Format: YYYY-MM-DD HH:MM:SS.")

@tool(
    description="Get the forecast information of a company",
    args_schema=CompanyForecastInput
)
async def company_forecast(market: str, symbol: str, period: str, trigger_time: str=None) -> str:
    if market == "CN-Stock":
        trigger_date = trigger_time.split(" ")[0].replace("-", "")
        if period < trigger_date:
            func_args = {
                "ts_code": symbol,
                "period": period
            }
            df = tushare_cached.run("forecast", func_kwargs=func_args)
            if df.empty:
                return {"error": f"No data found for forecast."}
            return df.to_markdown()
        else:
            return {"error": "The period is not in the trigger time."}
    else:
        return {"error": "Not supported yet."}


class CompanyExpressInput(BaseModel):
    market: str = Field(description="The market of the company.")
    symbol: str = Field(description="The symbol of the company.")
    period: str = Field(description="The period of the financial data. Format: YYYYMMDD. \
                        Reporting period (the last day of each quarter, for example, \
                        20171231 for annual report, 20170630 for semi-annual report, 20170930 \
                        for third quarter report).")
    trigger_time: str = Field(description="The trigger time of the financial data. Format: YYYY-MM-DD HH:MM:SS.")

@tool(
    description="Get the express information of a company",
    args_schema=CompanyExpressInput
)
async def company_express(market: str, symbol: str, period: str, trigger_time: str=None) -> str:
    if market == "CN-Stock":
        trigger_date = trigger_time.split(" ")[0].replace("-", "")
        if period < trigger_date:
            func_args = {
                "ts_code": symbol,
                "period": period
            }
            df = tushare_cached.run("express", func_kwargs=func_args)
            if df.empty:
                return {"error": f"No data found for express."}
            return df.to_markdown()
        else:
            return {"error": "The period is not in the trigger time."}
    else:
        return {"error": "Not supported yet."}


class CompanyDividendInput(BaseModel):
    market: str = Field(description="The market of the company.")
    symbol: str = Field(description="The symbol of the company.")
    period: str = Field(description="The period of the financial data. Format: YYYYMMDD. \
                        Reporting period (the last day of each quarter, for example, \
                        20171231 for annual report, 20170630 for semi-annual report, 20170930 \
                        for third quarter report).")
    trigger_time: str = Field(description="The trigger time of the financial data. Format: YYYY-MM-DD HH:MM:SS.")

@tool(
    description="Get the dividend information of a company",
    args_schema=CompanyDividendInput
)
async def company_dividend(market: str, symbol: str, period: str, trigger_time: str=None) -> str:
    if market == "CN-Stock":
        trigger_date = trigger_time.split(" ")[0].replace("-", "")
        if period < trigger_date:
            func_args = {
                "ts_code": symbol,
                "period": period
            }
            df = tushare_cached.run("dividend", func_kwargs=func_args)
            if df.empty:
                return {"error": f"No data found for dividend."}
            return df.to_markdown()
        else:
            return {"error": "The period is not in the trigger time."}
    else:
        return {"error": "Not supported yet."}


class CompanyFinaIndicatorInput(BaseModel):
    market: str = Field(description="The market of the company.")
    symbol: str = Field(description="The symbol of the company.")
    period: str = Field(description="The period of the financial data. Format: YYYYMMDD. \
                        Reporting period (the last day of each quarter, for example, \
                        20171231 for annual report, 20170630 for semi-annual report, 20170930 \
                        for third quarter report).")
    trigger_time: str = Field(description="The trigger time of the financial data. Format: YYYY-MM-DD HH:MM:SS.")

@tool(
    description="Get the financial indicator information of a company",
    args_schema=CompanyFinaIndicatorInput
)
async def company_fina_indicator(market: str, symbol: str, period: str, trigger_time: str=None) -> str:
    if market == "CN-Stock":
        trigger_date = trigger_time.split(" ")[0].replace("-", "")
        if period < trigger_date:
            func_args = {
                "ts_code": symbol,
                "period": period
            }
            df = tushare_cached.run("fina_indicator", func_kwargs=func_args)
            if df.empty:
                return {"error": f"No data found for fina_indicator."}
            return df.to_markdown()
        else:
            return {"error": "The period is not in the trigger time."}
    else:
        return {"error": "Not supported yet."}


class CompanyFinaAuditInput(BaseModel):
    market: str = Field(description="The market of the company.")
    symbol: str = Field(description="The symbol of the company.")
    period: str = Field(description="The period of the financial data. Format: YYYYMMDD. \
                        Reporting period (the last day of each quarter, for example, \
                        20171231 for annual report, 20170630 for semi-annual report, 20170930 \
                        for third quarter report).")
    trigger_time: str = Field(description="The trigger time of the financial data. Format: YYYY-MM-DD HH:MM:SS.")

@tool(
    description="Get the financial audit information of a company",
    args_schema=CompanyFinaAuditInput
)
async def company_fina_audit(market: str, symbol: str, period: str, trigger_time: str=None) -> str:
    if market == "CN-Stock":
        trigger_date = trigger_time.split(" ")[0].replace("-", "")
        if period < trigger_date:
            func_args = {
                "ts_code": symbol,
                "period": period
            }
            df = tushare_cached.run("fina_audit", func_kwargs=func_args)
            if df.empty:
                return {"error": f"No data found for fina_audit."}
            return df.to_markdown()
        else:
            return {"error": "The period is not in the trigger time."}
    else:
        return {"error": "Not supported yet."}


class CompanyFinaMainbzInput(BaseModel):
    market: str = Field(description="The market of the company.")
    symbol: str = Field(description="The symbol of the company.")
    period: str = Field(description="The period of the financial data. Format: YYYYMMDD. \
                        Reporting period (the last day of each quarter, for example, \
                        20171231 for annual report, 20170630 for semi-annual report, 20170930 \
                        for third quarter report).")
    trigger_time: str = Field(description="The trigger time of the financial data. Format: YYYY-MM-DD HH:MM:SS.")

@tool(
    description="Get the main business information of a company",
    args_schema=CompanyFinaMainbzInput
)
async def company_fina_mainbz(market: str, symbol: str, period: str, trigger_time: str=None) -> str:
    if market == "CN-Stock":
        trigger_date = trigger_time.split(" ")[0].replace("-", "")
        if period < trigger_date:
            func_args = {
                "ts_code": symbol,
                "period": period
            }
            df = tushare_cached.run("fina_mainbz", func_kwargs=func_args)
            if df.empty:
                return {"error": f"No data found for fina_mainbz."}
            return df.to_markdown()
        else:
            return {"error": "The period is not in the trigger time."}
    else:
        return {"error": "Not supported yet."}


class CompanyDisclosureDateInput(BaseModel):
    market: str = Field(description="The market of the company.")
    symbol: str = Field(description="The symbol of the company.")
    period: str = Field(description="The period of the financial data. Format: YYYYMMDD. \
                        Reporting period (the last day of each quarter, for example, \
                        20171231 for annual report, 20170630 for semi-annual report, 20170930 \
                        for third quarter report).")
    trigger_time: str = Field(description="The trigger time of the financial data. Format: YYYY-MM-DD HH:MM:SS.")

@tool(
    description="Get the disclosure date information of a company",
    args_schema=CompanyDisclosureDateInput
)
async def company_disclosure_date(market: str, symbol: str, period: str, trigger_time: str=None) -> str:
    if market == "CN-Stock":
        trigger_date = trigger_time.split(" ")[0].replace("-", "")
        if period < trigger_date:
            func_args = {
                "ts_code": symbol,
                "period": period
            }
            df = tushare_cached.run("disclosure_date", func_kwargs=func_args)
            if df.empty:
                return {"error": f"No data found for disclosure_date."}
            return df.to_markdown()
        else:
            return {"error": "The period is not in the trigger time."}
    else:
        return {"error": "Not supported yet."}


if __name__ == "__main__":
    result = asyncio.run(company_financial_info.ainvoke({"market": "US-Stock", "symbol": "AAPL", "task": "最近财报", "trigger_time": "2025-01-09 15:00:00"}))
    print(result)
    pass