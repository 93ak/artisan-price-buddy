import httpx
import re
import statistics
from bs4 import BeautifulSoup
from db.database import get_market_cache, set_market_cache

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-IN,en;q=0.9",
}

PRICE_RE = re.compile(r'(?:₹|Rs\.?\s*)(\d[\d,]*(?:\.\d{1,2})?)', re.IGNORECASE)

# Domains known to list individual handmade products at realistic prices
PREFERRED_DOMAINS = ["etsy.com", "meesho.com", "amazon.in", "flipkart.com",
                     "indiamart.com", "craftsvilla.com", "jaypore.com", "okhai.com"]

async def duckduckgo_search(query: str, num_results: int = 8) -> list[str]:
    search_query = f"{query} buy handmade India site:etsy.com OR site:meesho.com OR site:amazon.in OR site:indiamart.com"
    url = "https://html.duckduckgo.com/html/"
    async with httpx.AsyncClient(timeout=15.0, headers=HEADERS, follow_redirects=True) as client:
        resp = await client.post(url, data={"q": search_query})
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for a in soup.select("a.result__url"):
            href = a.get("href", "")
            if href and href.startswith("http") and not any(
                x in href for x in ["duckduckgo", "bing.com", "google.com"]
            ):
                links.append(href)
                if len(links) >= num_results:
                    break
        return links

def is_product_price(val: float, query: str) -> bool:
    """
    Heuristic: is this a plausible unit price for this product?
    Filters out page-level noise like course fees, bulk rates, unrelated numbers.
    """
    # Tight range: ₹50 to ₹15000 for most handmade items
    if val < 50 or val > 15000:
        return False
    # Round numbers above 5000 are suspicious (course fees, bulk minimums)
    if val >= 5000 and val % 1000 == 0:
        return False
    return True

async def scrape_prices_from_url(url: str, query: str = "") -> list[float]:
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=HEADERS, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, "html.parser")

            # Focus on price-bearing elements first (product cards, price tags)
            price_elements = soup.select(
                "[class*='price'], [class*='Price'], [class*='cost'], "
                "[class*='amount'], [itemprop='price'], [class*='mrp']"
            )
            targeted_text = " ".join(el.get_text() for el in price_elements)

            # Fall back to full page if no targeted elements found
            if not targeted_text.strip():
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                targeted_text = soup.get_text(separator=" ")

            matches = PRICE_RE.findall(targeted_text)
            prices = []
            seen = set()
            for m in matches:
                try:
                    val = float(m.replace(",", ""))
                    if val not in seen and is_product_price(val, query):
                        prices.append(val)
                        seen.add(val)
                except ValueError:
                    pass
            return prices[:8]
    except Exception:
        return []

def compute_market_stats(all_prices: list[float]) -> dict:
    if not all_prices:
        return {
            "market_min": None, "market_avg": None, "market_max": None,
            "market_median": None, "sample_count": 0,
            "note": "No market data found."
        }

    # IQR outlier removal — more aggressive (1.0× instead of 1.5×)
    if len(all_prices) >= 4:
        q1 = statistics.quantiles(all_prices, n=4)[0]
        q3 = statistics.quantiles(all_prices, n=4)[2]
        iqr = q3 - q1
        filtered = [p for p in all_prices if q1 - 1.0*iqr <= p <= q3 + 1.0*iqr]
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
        "note": f"Based on {len(filtered)} prices (outliers removed)"
    }

async def research_market(query: str) -> dict:
    cache_key = query.lower().strip()
    cached = get_market_cache(cache_key)
    if cached:
        cached["from_cache"] = True
        return cached

    all_prices = []
    urls = await duckduckgo_search(query)
    sources = []

    for url in urls[:5]:
        prices = await scrape_prices_from_url(url, query)
        if prices:
            all_prices.extend(prices)
            sources.append(url)

    stats = compute_market_stats(all_prices)
    stats["sources"] = sources
    stats["from_cache"] = False

    set_market_cache(cache_key, stats)
    return stats