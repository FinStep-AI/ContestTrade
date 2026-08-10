"""
sina news data crawler
"""
import asyncio
import aiohttp
import re
import json
import os
import time
from datetime import datetime
import random
import html
import pandas as pd
import sys
current_dir = os.path.dirname(__file__)
package_root = os.path.dirname(current_dir)
if package_root not in sys.path:
    sys.path.insert(0, package_root)
from data_source.data_source_base import DataSourceBase
from loguru import logger


class SinaNewsCrawl(DataSourceBase):
    def __init__(self, start_page=1, end_page=20):
        super().__init__("sina_news_crawl")
        self.start_page = start_page
        self.end_page = end_page
        # 使用你提供的完整URL格式，page/r/callback 将在请求时动态生成
        self.base_url = "http://feed.mix.sina.com.cn/api/roll/get"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
            "Referer": "https://finance.sina.com.cn/",
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        self.all_items = []
        self.fetch_full_intro = True  # 是否抓取文章页以补全 intro
        self.article_concurrency = 10 # 控制抓取文章页的并发
        
    async def fetch_page(self, session, page):
        """异步获取单个页面的数据"""
        params = {
            "pageid": 384,
            "lid": 2519,
            "k": "",
            "num": 50,
            "page": page
        }
        
        try:
            async with session.get(self.base_url, params=params, headers=self.headers, timeout=15) as response:
                text = await response.text()
                
                # 兼容 JSONP 与 纯 JSON
                m = re.search(r'^\s*[\w$]+\((.*)\)\s*;?\s*$', text.strip(), re.S)
                json_text = m.group(1) if m else text.strip()
                data = json.loads(json_text)
                
                # 提取items（仅保留指定字段）
                items = self.extract_items(data, page)

                # 尝试补全 intro
                if self.fetch_full_intro and items:
                    await self.enrich_items_with_full_intro(session, items)

                return items
                
        except Exception as e:
            return []
    
    def extract_items(self, data, page):
        """提取新闻items，并裁剪为目标字段集"""
        try:
            if isinstance(data, dict):
                result = data.get("result", {})
                if isinstance(result, dict):
                    data_field = result.get("data", [])
                    if isinstance(data_field, list):
                        processed_items = []
                        for raw in data_field:
                            if not isinstance(raw, dict):
                                continue
                            # 选取第一个可用的时间字段
                            candidate_keys = [
                                "ctime", "intime", "mtime", "create_time", "createtime",
                                "pub_time", "pubTime", "pubdate", "pubDate", "time", "update_time"
                            ]
                            raw_time_value = None
                            for key in candidate_keys:
                                if key in raw and raw.get(key) not in (None, ""):
                                    raw_time_value = raw.get(key)
                                    break
                            publish_time = self.normalize_publish_time(raw_time_value)

                            # 本地可用的简介
                            intro_local = self.choose_best_intro_local(raw)
                            # 目标URL（优先PC，其次WAP，其次urls数组）
                            url = self.choose_best_url(raw)

                            # 仅保留指定字段
                            processed_items.append({
                                "title": raw.get("title") or raw.get("stitle") or "",
                                "intro": intro_local or "",
                                "publish_time": publish_time,
                                "media_name": raw.get("media_name") or "",
                                "url": url or "",
                            })
                        return processed_items
            return []
        except Exception as e:
            print(f"第 {page} 页数据解析失败: {e}")
            return []
    
    def normalize_publish_time(self, raw_value):
        """将多种时间格式标准化为 'YYYY-MM-DD HH:MM:SS' 字符串"""
        try:
            if raw_value is None:
                return None
            # 数字时间戳（秒或毫秒）
            if isinstance(raw_value, (int, float)):
                timestamp = int(raw_value)
            elif isinstance(raw_value, str) and re.fullmatch(r"\d{10,13}", raw_value):
                timestamp = int(raw_value)
            else:
                # 尝试解析常见的时间字符串
                if isinstance(raw_value, str):
                    for fmt in [
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%d %H:%M",
                        "%Y-%m-%d",
                        "%Y/%m/%d %H:%M:%S",
                        "%Y/%m/%d %H:%M",
                        "%Y/%m/%d",
                        "%Y年%m月%d日 %H:%M",
                        "%Y年%m月%d日",
                    ]:
                        try:
                            dt = datetime.strptime(raw_value.strip(), fmt)
                            return dt.strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            pass
                # 无法识别则原样返回字符串
                return str(raw_value)

            # 毫秒与秒的区分
            if timestamp > 1_000_000_000_000:
                timestamp //= 1000
            elif 0 < timestamp < 10_000_000_000:
                pass
            else:
                # 非常规范围，保险起见取前10位
                timestamp = int(str(timestamp)[:10])

            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(raw_value)

    def choose_best_url(self, raw_item):
        """选择最合适的文章URL"""
        url = raw_item.get("url")
        if url:
            return url
        # 有些返回的 urls 是 JSON 字符串
        urls_field = raw_item.get("urls")
        if isinstance(urls_field, list) and urls_field:
            return urls_field[0]
        if isinstance(urls_field, str) and urls_field.strip().startswith("["):
            try:
                parsed = json.loads(urls_field)
                if isinstance(parsed, list) and parsed:
                    return parsed[0]
            except Exception:
                pass
        wapurl = raw_item.get("wapurl")
        if wapurl:
            return wapurl
        return None

    def choose_best_intro_local(self, raw_item):
        """在不请求文章页的情况下，选取最合适的简介字段"""
        candidates = [raw_item.get("intro"), raw_item.get("summary"), raw_item.get("wapsummary")]
        candidates = [c for c in candidates if isinstance(c, str) and c.strip()]
        if not candidates:
            return None
        # 选择最长的一条
        best = max(candidates, key=lambda x: len(x))
        return best

    def should_fetch_full_intro(self, intro_text):
        """判断是否需要抓取文章页补全简介"""
        if not intro_text:
            return True
        text = intro_text.strip()
        if len(text) < 60:
            return True
        if text.endswith("…") or text.endswith("..."):
            return True
        return False

    async def enrich_items_with_full_intro(self, session, items):
        """并发抓取文章页，补全 intro"""
        semaphore = asyncio.Semaphore(self.article_concurrency)

        async def process_one(item):
            if not self.should_fetch_full_intro(item.get("intro")):
                return
            url = item.get("url")
            if not url:
                return
            try:
                async with semaphore:
                    intro_full = await self.fetch_article_intro(session, url)
                if intro_full and len(intro_full) > len(item.get("intro") or ""):
                    item["intro"] = intro_full
            except Exception:
                pass

        await asyncio.gather(*[process_one(it) for it in items])

    async def fetch_article_intro(self, session, url):
        """抓取文章页简介：优先 meta description / og:description，其次正文首段"""
        try:
            async with session.get(url, headers=self.headers, timeout=15) as resp:
                html_text = await resp.text(errors="ignore")
            if not html_text:
                return None
            # 先尝试 meta description / og:description
            meta_desc = self._extract_meta_description(html_text)
            if meta_desc:
                return meta_desc
            # 退化到正文首段
            first_paragraph = self._extract_first_paragraph(html_text)
            if first_paragraph:
                return first_paragraph
            return None
        except Exception:
            return None

    def _extract_meta_description(self, html_text):
        """从HTML中提取<meta name="description">或<meta property="og:description">"""
        try:
            # name=description
            m1 = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', html_text, re.I | re.S)
            if m1:
                return html.unescape(self._clean_whitespace(m1.group(1)))
            # property=og:description
            m2 = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']', html_text, re.I | re.S)
            if m2:
                return html.unescape(self._clean_whitespace(m2.group(1)))
            return None
        except Exception:
            return None

    def _extract_first_paragraph(self, html_text):
        """从常见容器中提取首段文本（简易正则版）"""
        try:
            # 常见正文容器 id/class: artibody, article, content
            container_patterns = [
                r'<div[^>]+id=["\']artibody["\'][^>]*>(.*?)</div>',
                r'<article[^>]*>(.*?)</article>',
                r'<div[^>]+class=["\'][^"\']*(?:article|content)[^"\']*["\'][^>]*>(.*?)</div>',
            ]
            for pat in container_patterns:
                m = re.search(pat, html_text, re.I | re.S)
                if m:
                    inner = m.group(1)
                    # 找第一个<p>
                    p = re.search(r'<p[^>]*>(.*?)</p>', inner, re.I | re.S)
                    if p:
                        text = self._strip_html_tags(p.group(1))
                        return self._clean_whitespace(text)
            # 兜底：全局第一个<p>
            p = re.search(r'<p[^>]*>(.*?)</p>', html_text, re.I | re.S)
            if p:
                text = self._strip_html_tags(p.group(1))
                return self._clean_whitespace(text)
            return None
        except Exception:
            return None

    def _strip_html_tags(self, text):
        text = re.sub(r'<script[\s\S]*?</script>', ' ', text, flags=re.I)
        text = re.sub(r'<style[\s\S]*?</style>', ' ', text, flags=re.I)
        text = re.sub(r'<[^>]+>', ' ', text)
        return html.unescape(text)

    def _clean_whitespace(self, text):
        return re.sub(r'\s+', ' ', (text or '')).strip()
    
    async def crawl_all_pages(self):
        start_time = time.time()
        
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            for page in range(self.start_page, self.end_page + 1):
                task = self.fetch_page(session, page)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for page, result in enumerate(results, start=self.start_page):
                if isinstance(result, Exception):
                    print(f"第 {page} 页发生异常: {result}")
                elif isinstance(result, list):
                    self.all_items.extend(result)
        return self.all_items
    

    async def get_data(self, trigger_time: str) -> pd.DataFrame:
        self.all_items = []  # 清空累积的数据
        
        try:
            items = await self.crawl_all_pages()
        except Exception as e:
            logger.error(f"❌ Failed to crawl pages: {e}")
            # 即使爬取失败，也尝试返回空DataFrame而不是报错
            logger.info("⚠️ Returning empty DataFrame due to crawl failure")
            return pd.DataFrame(columns=['title', 'content', 'pub_time', 'url'])
        
        # 检查是否有数据
        if not items:
            logger.warning("⚠️ No items collected from crawling")
            return pd.DataFrame(columns=['title', 'content', 'pub_time', 'url'])
        
        logger.info(f"📊 Processing {len(items)} collected items...")
        
        df = pd.DataFrame(items)
        
        # 处理时间字段
        if not df.empty and 'publish_time' in df.columns:
            df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
            end_dt = pd.to_datetime(trigger_time, errors='coerce')
            mask = pd.Series(True, index=df.index)
            if not pd.isna(end_dt):
                start_dt = end_dt - pd.Timedelta(days=1)
                mask &= (df['publish_time'] >= start_dt) & (df['publish_time'] < end_dt)
            df = df.loc[mask].reset_index(drop=True)
            df['pub_time'] = df['publish_time'].dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            df['pub_time'] = ''

        if 'intro' in df.columns:
            df['content'] = df['intro'].apply(lambda x: self._clean_whitespace(self._strip_html_tags(str(x))))
        else:
            df['content'] = ""

        df['pub_time'] = df['pub_time'].fillna('')

        # 确保所有必需的列都存在
        keep_cols = ['title', 'content', 'pub_time', 'url']
        for col in keep_cols:
            if col not in df.columns:
                df[col] = ""

        df = df[keep_cols].copy()
        logger.info(f"get sina news until {trigger_time} success. Total {len(df)} rows")
        return df

if __name__ == "__main__":
    crawler = SinaNewsCrawl(start_page=1, end_page=50)
    df = asyncio.run(crawler.get_data("2025-08-21 15:00:00"))
    print(len(df))
    # try:
    #     output_path = os.path.join(os.path.dirname(__file__), "sina_news_crawl.json")
    #     df.to_json(output_path, orient="records", force_ascii=False, date_format="iso")
    #     print(f"Saved JSON to: {output_path}")
    # except Exception as e:
    #     print(f"Failed to save JSON: {e}")
 