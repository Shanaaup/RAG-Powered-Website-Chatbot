from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import asyncio
import os
import sys

OUTPUT_DIR = "scraped_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def scrape_website(url):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

        if result.success:
            html = result.cleaned_html
            soup = BeautifulSoup(html, "html.parser")

            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Extract only main content (Wikipedia specific)
            main_content = soup.find("main")

            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

            # Save as text file
            safe_name = url.replace("https://", "").replace("http://", "").replace("/", "_")
            filename = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")

            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)

            print("✅ Clean text saved successfully!")
            print("📁 File:", filename)

        else:
            print("❌ Failed:", result.error)


if __name__ == "__main__":
    url = input("Enter URL: ").strip()
    if not url.startswith("http"):
        url = "https://" + url

    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(scrape_website(url))
        loop.close()
    else:
        asyncio.run(scrape_website(url))