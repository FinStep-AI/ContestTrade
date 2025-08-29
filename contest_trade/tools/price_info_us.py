"""
Price Info Tools
"""
from pathlib import Path
import sys
import asyncio
import pandas as pd
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import asyncio
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from utils.market_manager import GLOBAL_MARKET_MANAGER
from tools.tool_utils import smart_tool


class PriceInfoInput(BaseModel):
    market: str = Field(description="The market of the company.")
    symbol: str = Field(description="The symbol of the company. Only one symbol is allowed.")
    trigger_time: str = Field(description="The trigger time of the financial data. Format: YYYY-MM-DD HH:MM:SS.")

@smart_tool(
    description="Get the price information of a symbol.",
    args_schema=PriceInfoInput,
    max_output_len=2000,
    timeout_seconds=3.0
)
async def price_info(market: str, symbol: str, trigger_time: str=None) -> str:
    triggle_date = trigger_time.split(" ")[0].replace("-", "")
    start_date = (datetime.strptime(triggle_date, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
    end_date = (datetime.strptime(triggle_date, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
    
    if market in ["CN-Stock", "US-Stock", "CSI300", "CSI500", "CSI1000"]:
        # recent K line info
        try:
            df = GLOBAL_MARKET_MANAGER.get_symbol_history_price(market, symbol, start_date, end_date)
            return {"result": df.to_markdown()}
        except Exception as e:
            print(str(e))
            return {"error": str(e)}
    else:
        return {"error": "Market not supported."}

if __name__ == "__main__":
    result = asyncio.run(price_info.ainvoke(
        { "market": "US-Stock", 
         "symbol": "AAPL", 
         "trigger_time": "2025-07-09 15:00:00"}))
    print(result)

    result = asyncio.run(price_info.ainvoke(
        { "market": "US-Stock", 
         "symbol": "TSLA", 
         "trigger_time": "2025-07-09 15:00:00"}))
    print(result)
