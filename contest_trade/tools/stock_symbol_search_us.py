from pathlib import Path
import sys
import asyncio
import pandas as pd
from typing import List, Dict, Any
from functools import lru_cache
import re

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from utils.market_manager import GLOBAL_MARKET_MANAGER
from tools.tool_utils import smart_tool

class StockSymbolSearchInput(BaseModel):
    market: str = Field(description="The target market. e.g., CN-Stock, US-Stock, HK-Stock, CN-ETF")
    queries: List[str] = Field(description="List of search queries: company names or stock symbols (partial match supported)")
    trigger_time: str = Field(description="The trigger time. Format: YYYY-MM-DD HH:MM:SS")
    limit_per_query: int = Field(default=5, description="Maximum number of results per query")
    match_mode: str = Field(default="best", description="Match mode: 'best' (top match), 'all' (all matches), 'exact' (exact only)")

def get_market_symbols_cached(market: str, trigger_date: str) -> pd.DataFrame:
    """Cache market symbols to avoid repeated API calls"""
    return GLOBAL_MARKET_MANAGER.get_market_symbols(market, f"{trigger_date} 15:00:00", full_market=True)

def calculate_match_score(query: str, ts_code: str, name: str) -> tuple[str, float]:
    """Calculate match score and type"""
    query_lower = query.lower()
    ts_code_lower = ts_code.lower()
    name_lower = name.lower()
    
    # Exact match
    if query == ts_code or query == name:
        return "exact", 1.0
    if query_lower == ts_code_lower or query_lower == name_lower:
        return "exact", 1.0
    
    # Prefix match
    if ts_code_lower.startswith(query_lower) or name_lower.startswith(query_lower):
        return "prefix", 0.9
    
    # Contains match
    if query_lower in ts_code_lower or query_lower in name_lower:
        return "contains", 0.8
    
    # Fuzzy match for partial code
    if re.search(re.escape(query_lower), ts_code_lower) or re.search(re.escape(query_lower), name_lower):
        return "fuzzy", 0.7
    
    return "none", 0.0

def search_single_query(symbols_df: pd.DataFrame, query: str, limit: int, match_mode: str, market: str) -> List[Dict[str, Any]]:
    """Search for a single query"""
    results = []
    
    for _, row in symbols_df.iterrows():
        ts_code = row.get('ts_code', '')
        name = row.get('name', '')
        
        match_type, score = calculate_match_score(query, ts_code, name)
        
        if match_mode == "exact" and match_type != "exact":
            continue
        
        if score > 0:
            results.append({
                "ts_code": ts_code,
                "name": name,
                "market": market,
                "match_type": match_type,
                "match_score": score
            })
    
    # Sort by score and match type priority
    results.sort(key=lambda x: (x['match_score'], x['match_type'] == 'exact'), reverse=True)
    
    if match_mode == "best":
        return results[:1]
    else:
        return results[:limit]

@smart_tool(
    description="Search for stock symbols by company names or partial symbols in batch mode.",
    args_schema=StockSymbolSearchInput,
    max_output_len=4000,
    timeout_seconds=3.0
)
async def stock_symbol_search(
    market: str, 
    queries: List[str], 
    trigger_time: str,
    limit_per_query: int = 5,
    match_mode: str = "best"
) -> Dict[str, Any]:
    
    try:
        trigger_date = trigger_time.split(" ")[0].replace("-", "")
        
        symbols_df = get_market_symbols_cached(market, trigger_date)
        
        if symbols_df.empty:
            return {
                "error": f"No symbols found for market {market} at {trigger_time}",
                "results": {},
                "summary": {"total_queries": len(queries), "successful_matches": 0, "failed_matches": len(queries)},
                "failed_queries": [{"query": q, "error": "No market data"} for q in queries]
            }
        
        # Process each query
        results = {}
        failed_queries = []
        
        for query in queries:
            try:
                matches = search_single_query(symbols_df, query, limit_per_query, match_mode, market)
                if matches:
                    results[query] = matches
                else:
                    failed_queries.append({"query": query, "error": "No matches found"})
            except Exception as e:
                failed_queries.append({"query": query, "error": str(e)})
        
        # Generate summary
        summary = {
            "total_queries": len(queries),
            "successful_matches": len(results),
            "failed_matches": len(failed_queries),
            "total_results": sum(len(matches) for matches in results.values())
        }
        
        return {
            "results": results,
            "summary": summary,
            "failed_queries": failed_queries,
            "market": market,
            "trigger_time": trigger_time
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "results": {},
            "summary": {"total_queries": len(queries), "successful_matches": 0, "failed_matches": len(queries)},
            "failed_queries": [{"query": q, "error": str(e)} for q in queries]
        }

if __name__ == "__main__":
    pass