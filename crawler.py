import os
import sys
import psutil
import asyncio
import requests
from xml.etree import ElementTree
import torch
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# ✅ Force NVIDIA GPU usage
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first NVIDIA GPU
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No GPU found. Running on CPU.")

# ✅ Output file
OUTPUT_FILE = "test.txt"
Crawled_Data = ""

async def crawl_parallel(urls: List[str], max_concurrent: int = 3):
    global Crawled_Data
    """Crawls multiple URLs in parallel, stores results in a file, and tracks memory usage."""
    print("\n=== Parallel Crawling with Browser Reuse + Memory Check ===")

    peak_memory = 0
    process = psutil.Process(os.getpid())

    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss  # in bytes
        if current_mem > peak_memory:
            peak_memory = current_mem
        print(f"{prefix} Current Memory: {current_mem // (1024 * 1024)} MB, Peak: {peak_memory // (1024 * 1024)} MB")

    # ✅ Minimal browser config
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # ✅ Create crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        results = []
        success_count = 0
        fail_count = 0

        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            tasks = [crawler.arun(url, config=crawl_config, session_id=f"session_{i+j}") for j, url in enumerate(batch)]

            # ✅ Log memory usage before crawling
            log_memory(prefix=f"Before batch {i // max_concurrent + 1}: ")

            # ✅ Fetch results concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # ✅ Log memory usage after crawling
            log_memory(prefix=f"After batch {i // max_concurrent + 1}: ")

            # ✅ Process results
            for url, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    print(f"❌ Error crawling {url}: {result}")
                    fail_count += 1
                elif hasattr(result, "success") and result.success:  # Ensure result has 'success' attribute
                    results.append(result[0].html) 
                    success_count += 1
                else:
                    fail_count += 1

        # ✅ Save results to file properly
        # with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        #     f.writelines(results)
        Crawled_Data = results
        print(f"\n✅ Crawling completed! Results saved to `{OUTPUT_FILE}`")
        print(f"✅ Successfully crawled: {success_count}, ❌ Failed: {fail_count}")

    finally:
        await crawler.close()
        log_memory(prefix="Final: ")
        print(f"📌 Peak memory usage: {peak_memory // (1024 * 1024)} MB")
        


def get_sitemap_urls(siteurl):
    """Fetches URLs from a sitemap XML file."""
    sitemap_url = siteurl+"sitemap.xml"

    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        return [loc.text for loc in root.findall('.//ns:loc', namespace)]
    except Exception as e:
        print(f"❌ Error fetching sitemap: {e}")
        return []

# SITE_URL = "https://www.linkedin.com/in/vinaychoudhary7525/"

async def Crawler(URL):
    await main(URL)
    return Crawled_Data

async def main(url):
    urls = get_sitemap_urls(url)
    if urls:
        print(f"🔍 Found {len(urls)} URLs to crawl")
        await crawl_parallel(urls, max_concurrent=10)
    else:
        print("⚠️ No URLs found to crawl")
        print(f"Crawling base URL {url}")
        await crawl_parallel([url])


if __name__ == "__main__":
    asyncio.run(main())
