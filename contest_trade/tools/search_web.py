"""
Search Web Tools
use bocha and serpapi to search web
"""
import sys
import os
import asyncio
import requests
import textwrap
from pathlib import Path
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from loguru import logger
import json
from config.config import cfg
from tools.tool_utils import smart_tool


sys.path.append(str(Path(__file__).parent.parent.resolve()))

def ask_bocha(payload: dict, BOCHA_API_KEY: str) -> list:
    """
    Performs a search using the Bocha AI API.
    API Key must be provided as an argument.
    """
    BOCHA_URL = "https://api.bochaai.com/v1/web-search"
    headers = {
        'Authorization': 'Bearer ' + BOCHA_API_KEY,
        'Content-Type': 'application/json'
    }

    start_date = payload.get("start_date")
    end_date = payload.get("end_date")

    try:
        start_formatted = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end_formatted = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        freshness = f"{start_formatted}..{end_formatted}"
    except (TypeError, IndexError):
        logger.error("Bocha search failed: start_date or end_date are missing or malformed in payload.")
        return []

    bocha_payload = {
        "query": payload.get("query", ""),
        "count": payload.get("topk", 3),
        "freshness": freshness
    }

    try:
        response = requests.post(BOCHA_URL, headers=headers, json=bocha_payload, timeout=5)
        response.raise_for_status()  # For non-200 responses

        response_data = response.json()
        web_pages = response_data.get("data", {}).get("webPages", {})
        values = web_pages.get("value", [])

        standardized_results = []
        for item in values[:bocha_payload["count"]]:
            standardized_results.append({
                "title": item.get("name", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("url", ""),
                "time": item.get("dateLastCrawled", "")[:10]
            })
        return standardized_results

    except requests.exceptions.RequestException as e:
        logger.error(f"Bocha API request failed: {e}")
        return []


def ask_google(payload: dict, SERP_API_KEY: str) -> list:
    """
    Performs a search using the SerpAPI (Google Search).
    API Key must be provided as an argument.
    """
    SERP_URL = "https://google.serper.dev/search"

    try:
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")
        headers = {
            "X-API-KEY": SERP_API_KEY,
            'Content-Type': 'application/json'
        }
        # 默认面向美股资讯，使用美国地区/英文结果；如需中文可在query中加入中文关键词
        params = {
            "q": payload.get("query", ""),
            "num": min(payload.get("topk", 3), 10),
            "gl": "us",
            "hl": "en"
        }

        if start_date and end_date:
            start_formatted = f"{int(start_date[4:6])}/{int(start_date[6:8])}/{start_date[:4]}"
            end_formatted = f"{int(end_date[4:6])}/{int(end_date[6:8])}/{end_date[:4]}"
            params["tbs"] = f"cdr:1,cd_min:{start_formatted},cd_max:{end_formatted}"

        payload = json.dumps(params)
        response = requests.request("POST", SERP_URL, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()

        standardized_results = []
        organic_results = data.get("organic", [])
        for item in organic_results[:params["num"]]:
            standardized_results.append({
                "title": item.get("title", ""), 
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""), 
                "time": item.get("date", "")
            })
        
        print(standardized_results)
        return standardized_results
    except requests.exceptions.RequestException as e:
        logger.error(f"SerpAPI request failed: {e}")
        return []



def build_search_result_context(results: list) -> str:
    """Formats a list of search results into a single string context."""
    if not results: return ""
    SearchResultFormat = textwrap.dedent("""
    <search_result id={id}>
    <title>{title}</title>
    <time>{time}</time>
    <url>{url}</url>
    <summary>{summary}</summary>
    </search_result>
    """)
    result_context = [
        SearchResultFormat.format(
            id=idx + 1, title=res.get('title', 'N/A'),
            summary=res.get('snippet', 'N/A'), url=res.get('url', 'N/A'),
            time=res.get('time', 'N/A')
        ) for idx, res in enumerate(results)
    ]
    return "\n".join(result_context)


class SearchWebInput(BaseModel):
    query: str = Field(description="The search keywords, separate with empty space. Simple specific keywords. No more than 3 keywords.")
    topk: int = Field(default=3, description="The number of top results to return, default is 3")
    trigger_time: str = Field(description="The trigger time of the search. Format: YYYY-MM-DD HH:MM:SS.")


@smart_tool(
    description="Searches information from the web based on a query and returns a list of results up to the specified limit.",
    args_schema=SearchWebInput,
    max_output_len=2000,
    timeout_seconds=10.0
)
async def search_web(query: str, topk: int = 3, trigger_time: str = None):
    """
    Main tool function that orchestrates the search process with a fallback mechanism.
    This tool's signature is compatible with the Pydantic model for LangChain.
    """
    if not trigger_time:
        logger.error("Search failed: 'trigger_time' is a mandatory parameter for this tool.")
        return ""

    trigger_date = trigger_time.split(" ")[0].replace("-", "")
    start_time = (datetime.strptime(trigger_date, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
    end_time = (datetime.strptime(trigger_date, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")

    payload = {
        "query": query, "topk": min(topk, 20),
        "start_date": start_time, "end_date": end_time
    }
    
    response = []
    serp_api_key = cfg.serp_key
    bocha_api_key = cfg.bocha_key

    if not serp_api_key and not bocha_api_key:
        logger.warning("No search API keys (SERP_API_KEY, BOCHA_API_KEY) are configured.")
        return ""

    # Priority 1: Try Google Search
    if serp_api_key:
        logger.info(f"Attempting search with Google (SerpAPI) for query: '{query}'")
        response = ask_google(payload, serp_api_key)
    
    # Priority 2: Fallback to Bocha if the first attempt fails
    if not response and bocha_api_key:
        logger.warning("Google search failed or was not configured. Falling back to Bocha AI.")
        logger.info(f"Attempting search with Bocha AI for query: '{query}'")
        response = ask_bocha(payload, bocha_api_key)
    
    if not response:
        logger.warning("All configured search providers failed to return results.")
        return ""
        
    logger.info(f"Search successful. Returning {len(response)} results.")
    return build_search_result_context(response)

if __name__ == "__main__":
    result = asyncio.run(search_web.ainvoke({"query": "最近电影", "topk": 3, "trigger_time": "2025-01-09 15:00:00"}))
    print(result)
