import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RecursiveWebScraper:
    def __init__(self, start_url, max_depth=2, max_pages=20):
        """
        Initialize the scraper with a starting URL.
        
        Args:
            start_url: URL to start scraping from.
            max_depth: How deep to follow links recursively.
            max_pages: Total limit of pages to scrape to prevent infinite loops.
        """
        self.start_url = start_url
        self.base_domain = urlparse(start_url).netloc
        self.max_depth = max_depth
        self.max_pages = max_pages
        
        self.visited = set()
        self.scraped_data = [] # List of dicts: {'url': url, 'content': text}

    def _is_valid_url(self, url):
        """Check if URL belongs to the same domain and is valid."""
        try:
            parsed = urlparse(url)
            # Only scrape same domain, ignore fragments or purely internal/javascript links
            if parsed.netloc and parsed.netloc != self.base_domain:
                return False
            if url.startswith(('javascript:', 'mailto:', 'tel:')):
                return False
            return True
        except Exception:
            return False

    def scrape(self, url=None, current_depth=0):
        """Recursively scrapes the URL and its internal links."""
        url = url or self.start_url
        
        if current_depth > self.max_depth or len(self.visited) >= self.max_pages:
            return

        # Normalize URL to avoid duplicates (strip fragments)
        url = url.split('#')[0]
        if url in self.visited:
            return
            
        self.visited.add(url)
        logger.info(f"Scraping ({current_depth}/{self.max_depth}): {url}")
        
        try:
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0 RAG-Bot'})
            response.raise_for_status()
            
            # Use BeautifulSoup to parse HTML depending on content type
            if 'text/html' not in response.headers.get('Content-Type', ''):
                logger.info(f"Skipping non-HTML page: {url}")
                return
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts, styles, and navigational elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
                
            # Extract text
            text_content = soup.get_text(separator=' ', strip=True)
            if text_content:
                self.scraped_data.append({
                    'url': url,
                    'content': text_content
                })
                
            # Find all links on the page if depth hasn't been reached
            if current_depth < self.max_depth:
                for link in soup.find_all('a', href=True):
                    next_url = urljoin(url, link['href'])
                    if self._is_valid_url(next_url):
                        self.scrape(next_url, current_depth + 1)
                        
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")

    def get_data(self):
        """Returns the accumulated scraped data."""
        if not self.visited:
            self.scrape()
        return self.scraped_data

if __name__ == "__main__":
    # Simple test
    test_url = "https://example.com"
    scraper = RecursiveWebScraper(test_url, max_depth=1, max_pages=3)
    data = scraper.get_data()
    print(f"Scraped {len(data)} pages.")
    for page in data:
        print(f"URL: {page['url']} | Content snippet: {page['content'][:100]}...")
