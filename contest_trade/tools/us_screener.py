"""
US Stock Screener Tool
基于 FMP 的 /stock-screener 接口进行美股选股。

输入: 一组可选过滤条件(filters)与limit。
支持的过滤键:
- exchange: NASDAQ/NYSE/AMEX
- sector: 例如 Technology, Financial Services
- industry: 例如 Semiconductors
- marketCapMin/marketCapMax (美元)
- priceMin/priceMax (美元)
- peMin/peMax
- dividendMin (股息率%)
- isEtf: true/false
"""
# Allow running this file directly via `python tools/us_screener.py`
if __name__ == "__main__" and __package__ is None:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from tools.tool_utils import smart_tool
from utils.fmp_utils import fmp_cached


class USStockScreenerInput(BaseModel):
    filters: Optional[Dict[str, Any]] = Field(default=None, description="筛选条件字典，见文档支持的键")
    limit: int = Field(default=50, description="返回数量上限，默认50，最大200")
    trigger_time: str = Field(description="触发时间 YYYY-MM-DD HH:MM:SS")


def _map_filters_to_fmp_params(filters: Dict[str, Any]) -> Dict[str, Any]:
    params = {}
    if not filters:
        return params

    # 直接映射字段
    direct_keys = ["exchange", "sector", "industry", "isEtf"]
    for k in direct_keys:
        if k in filters and filters[k] is not None:
            params[k] = filters[k]

    # 数值范围映射
    if "marketCapMin" in filters:
        params["marketCapMoreThan"] = filters["marketCapMin"]
    if "marketCapMax" in filters:
        params["marketCapLowerThan"] = filters["marketCapMax"]

    if "priceMin" in filters:
        params["priceMoreThan"] = filters["priceMin"]
    if "priceMax" in filters:
        params["priceLowerThan"] = filters["priceMax"]

    # if "peMin" in filters:
    #     params["peMoreThan"] = filters["peMin"]
    # if "peMax" in filters:
    #     params["peLowerThan"] = filters["peMax"]

    if "dividendMin" in filters:
        params["dividendMoreThan"] = filters["dividendMin"]

    return params


@smart_tool(
    description="US stock screener based on FMP /stock-screener API. 返回满足过滤条件的美股列表。",
    args_schema=USStockScreenerInput,
    max_output_len=4000,
    timeout_seconds=8.0,
)
async def us_stock_screener(filters: Optional[Dict[str, Any]] = None, limit: int = 50, trigger_time: str = None):
    try:
        params = _map_filters_to_fmp_params(filters or {})
        # FMP 官方案例接口: /stock-screener
        result = fmp_cached.run('/stock-screener', params, verbose=False)
        if not result:
            return {"results": [], "count": 0}
        if limit:
            result = result[:min(200, max(1, limit))]
        # 仅保留关键字段
        keys = ["symbol", "companyName", "price", "marketCap", "sector", "industry", "beta", "volume", "exchange"]
        cleaned = []
        for item in result:
            cleaned.append({k: item.get(k) for k in keys})
        return {"results": cleaned, "count": len(cleaned), "filters": params}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import asyncio
    async def main():
        r = await us_stock_screener.ainvoke({
            "filters": {"exchange": "NASDAQ", "marketCapMin": 5_000_000_000},
            "limit": 20,
            "trigger_time": "2025-08-12 09:00:00"
        })
        print(r)
    asyncio.run(main())


