"""
thx news data crawler
"""
import asyncio
import requests
import json
import re
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import time
import random
import os
import sys
from loguru import logger
current_dir = os.path.dirname(__file__)
package_root = os.path.dirname(current_dir)
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from data_source.data_source_base import DataSourceBase
from crawl4ai import *


class ThxNewsCrawl(DataSourceBase):
    def __init__(self, max_pages: int = 5, enable_frontend_crawl: bool = True):
        super().__init__("thx_news_crawl")
        self.max_pages = max_pages
        self.enable_frontend_crawl = enable_frontend_crawl

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def parse_pub_time_from_frontend(self, line: str, url: str) -> str:
        try:
            date_part = None
            m_url_date = re.search(r"/(\d{8})/", url or "")
            if m_url_date:
                ymd = m_url_date.group(1)
                date_part = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
            time_part = None
            m_time = re.search(r"\b(\d{1,2}):(\d{2})\b", line or "")
            if m_time:
                h = int(m_time.group(1))
                m = int(m_time.group(2))
                if 0 <= h <= 23 and 0 <= m <= 59:
                    time_part = f"{h:02d}:{m:02d}:00"
            if not date_part:
                m_cn = re.search(r"(\d{4})[年/\\-](\d{1,2})[月/\\-](\d{1,2})", line or "")
                if m_cn:
                    y = int(m_cn.group(1))
                    mo = int(m_cn.group(2))
                    d = int(m_cn.group(3))
                    date_part = f"{y:04d}-{mo:02d}-{d:02d}"
            if date_part and time_part:
                return f"{date_part} {time_part}"
            if date_part:
                return f"{date_part} 00:00:00"
            return ""
        except Exception:
            return ""

    def extract_company_news_from_markdown(self, md: str) -> List[Dict[str, Any]]:
        records = []
        for raw_line in (md or "").splitlines():
            line = raw_line.strip()
            if not (line.startswith('* ') or line.startswith('- ')):
                continue
            m_link = re.search(r"\[([^\]]+)\]\((https?://[^)\s]+)[^)]*\)", line)
            if not m_link:
                continue
            title = (m_link.group(1) or "").strip()
            url = (m_link.group(2) or "").strip()
            if not re.search(r"/(\d{8})/", url):
                continue
            tail = line[m_link.end():]
            m_intro = re.search(r"\[([^\]]+)\]", tail)
            intro = (m_intro.group(1) or "").strip() if m_intro else ""
            content = self.clean_text(intro)
            pub_time = self.parse_pub_time_from_frontend(line, url)
            records.append({
                "title": title or "",
                "content": content or "",
                "pub_time": pub_time or "",
                "url": url or "",
            })
        return records

    def clean_html_content(self, html_content: str) -> str:
        if not html_content:
            return ""
        clean_text = re.sub(r'<[^>]+>', '', html_content)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        clean_text = clean_text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        return clean_text

    def parse_pub_time(self, timestamp: int) -> str:
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return ""

    def get_news_data(self, page: int = 1, pagesize: int = 400) -> List[Dict[str, Any]]:
        url = "https://news.10jqka.com.cn/tapp/news/push/stock/"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Referer': 'https://news.10jqka.com.cn/realtimenews.html',
            'Origin': 'https://news.10jqka.com.cn',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'X-Requested-With': 'XMLHttpRequest'
        }
        
        params = {
            'page': page,
            'tag': '',
            'track': 'website',
            'pagesize': pagesize
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            news_list = data.get('data', {}).get('list', [])
            processed_news = []
            for news in news_list:
                title = news.get('title', '')
                content = self.clean_html_content(news.get('digest', '')) 
                pub_time = self.parse_pub_time(int(news.get('ctime', 0))) 
                news_url = news.get('url', '')
                
                if not news_url and news.get('id'):
                    news_url = f"https://news.10jqka.com.cn/tapp/news/push/stock/{news.get('id')}/"
                
                processed_news.append({
                    'title': title,
                    'content': content,
                    'pub_time': pub_time,
                    'url': news_url
                })
            
            return processed_news
            
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return []
        except Exception as e:
            print(f"未知错误: {e}")
            return []

    def crawl_multiple_pages(self) -> List[Dict[str, Any]]:
        all_news = []
        
        for page in range(1, self.max_pages + 1):
            page_news = self.get_news_data(page=page, pagesize=400)
            
            if not page_news:
                break
                
            all_news.extend(page_news)
            if page < self.max_pages:
                delay = random.uniform(1, 3)
                time.sleep(delay)
        
        return all_news

    async def crawl_frontend_pages(self) -> List[Dict[str, Any]]:
        if not self.enable_frontend_crawl:
            logger.info("Frontend crawling is disabled")
            return []
        
        try:
            async with AsyncWebCrawler() as crawler:
                company_news_urls = [
                    "https://stock.10jqka.com.cn/companynews_list/index.shtml",
                    *[f"https://stock.10jqka.com.cn/companynews_list/index_{i}.shtml" for i in range(2, 21)],
                ]
                
                hsdp_urls = [
                    "https://stock.10jqka.com.cn/hsdp_list/index.shtml",
                    *[f"https://stock.10jqka.com.cn/hsdp_list/index_{i}.shtml" for i in range(2, 21)],
                ]
                
                # Combine all URLs
                page_urls = company_news_urls + hsdp_urls
                logger.info(f"Starting frontend crawl for {len(page_urls)} pages")
                
                results = await crawler.arun_many(urls=page_urls)
                logger.info(f"✅ Frontend crawl API calls completed, processing {len(results or [])} responses")

                # Merge records from all pages
                all_records = []
                page_records = []
                for i, res in enumerate(results or []):
                    page_markdown = getattr(res, "markdown", "")
                    page_records_count = len(self.extract_company_news_from_markdown(page_markdown))
                    page_records.append(page_records_count)
                    all_records.extend(
                        self.extract_company_news_from_markdown(page_markdown)
                    )
                
                logger.info(f"Frontend crawl completed, got {len(all_records)} records")
                return all_records
                
        except Exception as e:
            logger.error(f"❌ Frontend crawling failed: {e}")
            return []

    async def get_data(self, trigger_time: str) -> pd.DataFrame:
        tasks = [
            asyncio.to_thread(self.crawl_multiple_pages),  # API爬取
            self.crawl_frontend_pages()  # 前端爬取
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理API爬取结果
        api_news_data = []
        if isinstance(results[0], list):
            api_news_data = results[0]
            logger.info(f"✅ API crawling completed successfully: {len(api_news_data)} records")
        elif isinstance(results[0], Exception):
            logger.error(f"❌ API crawling failed: {results[0]}")
            logger.info("⚠️ Will continue with frontend data only...")
        else:
            logger.warning(f"⚠️ API crawling returned unexpected type: {type(results[0])}")
        
        # 处理前端爬取结果
        frontend_news_data = []
        if isinstance(results[1], list):
            frontend_news_data = results[1]
            logger.info(f"✅ Frontend crawling completed successfully: {len(frontend_news_data)} records")
        elif isinstance(results[1], Exception):
            logger.error(f"❌ Frontend crawling failed: {results[1]}")
            logger.info("⚠️ Will continue with API data only...")
        else:
            logger.warning(f"⚠️ Frontend crawling returned unexpected type: {type(results[1])}")
        
        # 合并所有数据
        all_news_data = api_news_data + frontend_news_data
        
        logger.info("📈 Data collection summary:")
        logger.info(f"  - API records: {len(api_news_data)}")
        logger.info(f"  - Frontend records: {len(frontend_news_data)}")
        logger.info(f"  - Combined total: {len(all_news_data)}")
        
        # 检查是否至少有一个数据源成功
        if not api_news_data and not frontend_news_data:
            error_msg = "❌ Both API and frontend crawling failed - no data available"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        elif not all_news_data:
            logger.warning("⚠️ No data collected from any source")
            return pd.DataFrame(columns=['title', 'content', 'pub_time', 'url'])
        
        seen_urls = set()
        deduped_news = []
        for news in all_news_data:
            url = news.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped_news.append(news)
        
        df = pd.DataFrame(deduped_news)
        
        if not df.empty and 'pub_time' in df.columns:
            df['pub_time'] = pd.to_datetime(df['pub_time'], errors='coerce')
            
            end_dt = pd.to_datetime(trigger_time, errors='coerce')
            
            if not pd.isna(end_dt):
                start_dt = end_dt - pd.Timedelta(days=1)
                mask = (df['pub_time'] >= start_dt) & (df['pub_time'] < end_dt)
                df = df.loc[mask].reset_index(drop=True)
                logger.info(f"时间筛选后剩余 {len(df)} 条数据（区间: {start_dt} 至 {end_dt}）")
            else:
                logger.warning(f"无法解析时间: {trigger_time}")
        
        df['pub_time'] = df['pub_time'].fillna('')
        if 'pub_time' in df.columns:
            df['pub_time'] = df['pub_time'].astype(str)
        
        keep_cols = ['title', 'content', 'pub_time', 'url']
        for col in keep_cols:
            if col not in df.columns:
                df[col] = ""
        
        df = df[keep_cols].copy()
        
        # 显示最终数据来源统计
        if api_news_data and frontend_news_data:
            logger.info(f"🎉 Successfully collected data from both sources until {trigger_time}")
        elif api_news_data:
            logger.info(f"🎉 Successfully collected data from API only until {trigger_time}")
        elif frontend_news_data:
            logger.info(f"🎉 Successfully collected data from frontend only until {trigger_time}")
        
        logger.info(f"Final result: {len(df)} rows after deduplication and time filtering")
        return df


if __name__ == "__main__":
    crawler = ThxNewsCrawl(max_pages=5, enable_frontend_crawl=True)
    df = asyncio.run(crawler.get_data("2025-08-21 15:00:00"))
    print(f"Total records: {len(df)}")
    print(df.head())
    # try:
    #     output_path = os.path.join(os.path.dirname(__file__), "10jqk_news_crawl.json")
    #     df.to_json(output_path, orient="records", force_ascii=False, date_format="iso")
    #     print(f"Saved JSON to: {output_path}")
    # except Exception as e:
    #     print(f"Failed to save JSON: {e}")



