import httpx
import re
import statistics
from bs4 import BeautifulSoup
from typing import Optional
from db.database import get_market_cache, set_market_cache

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-IN,en;q=0.9",
}

# INR price pattern: ₹ or Rs followed by digits
PRICE_RE = re.compile(r'(?:₹|Rs\.?\s*)(\d[\d,]*(?:\.\d{1,2})?)', re.IGNORECASE)

async def duckduckgo_search(query: str, num_results: int = 8) -> list[str]:
    """Search DuckDuckGo and return result URLs."""
    search_query = f"{query} buy price India handmade"
    url = "https://html.duckduckgo.com/html/"
    async with httpx.AsyncClient(timeout=15.0, headers=HEADERS, follow_redirects=True) as client:
        resp = await client.post(url, data={"q": search_query})
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for a in soup.select("a.result__url"):
            href = a.get("href", "")
            if href and href.startswith("http") and not any(x in href for x in ["duckduckgo", "bing.com", "google.com"]):
                links.append(href)
                if len(links) >= num_results:
                    break
        return links

async def scrape_prices_from_url(url: str) -> list[float]:
    """Scrape INR prices from a product page."""
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=HEADERS, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove scripts and styles
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = soup.get_text(separator=" ")
            matches = PRICE_RE.findall(text)
            prices = []
            for m in matches:
                try:
                    val = float(m.replace(",", ""))
                    # Filter: only plausible handmade product prices (₹50–₹50000)
                    if 50 <= val <= 50000:
                        prices.append(val)
                except ValueError:
                    pass
            return prices[:10]  # cap per page
    except Exception:
        return []

def compute_market_stats(all_prices: list[float]) -> dict:
    if not all_prices:
        return {
            "market_min": None, "market_avg": None, "market_max": None,
            "market_median": None, "sample_count": 0,
            "note": "No market data found. Estimates based on typical market range."
        }
    # Remove outliers (IQR method)
    if len(all_prices) >= 4:
        q1 = statistics.quantiles(all_prices, n=4)[0]
        q3 = statistics.quantiles(all_prices, n=4)[2]
        iqr = q3 - q1
        filtered = [p for p in all_prices if q1 - 1.5*iqr <= p <= q3 + 1.5*iqr]
    else:
        filtered = all_prices

    if not filtered:
        filtered = all_prices

    return {
        "market_min": round(min(filtered)),
        "market_avg": round(statistics.mean(filtered)),
        "market_median": round(statistics.median(filtered)),
        "market_max": round(max(filtered)),
        "sample_count": len(filtered),
        "note": f"Based on {len(filtered)} scraped prices (outliers removed)"
    }

async def research_market(query: str) -> dict:
    """Full pipeline: search → scrape → stats. Returns cached if fresh."""
    cache_key = query.lower().strip()
    cached = get_market_cache(cache_key)
    if cached:
        cached["from_cache"] = True
        return cached

    all_prices = []
    urls = await duckduckgo_search(query)
    sources = []

    for url in urls[:5]:  # limit to 5 pages
        prices = await scrape_prices_from_url(url)
        if prices:
            all_prices.extend(prices)
            sources.append(url)

    stats = compute_market_stats(all_prices)
    stats["sources"] = sources
    stats["from_cache"] = False

    set_market_cache(cache_key, stats)
    return stats
