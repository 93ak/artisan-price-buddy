import httpx
import re
import statistics
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from db.database import get_market_cache, set_market_cache

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-IN,en;q=0.9",
}

PRICE_RE = re.compile(r'(?:₹|Rs\.?\s*)(\d[\d,]*(?:\.\d{1,2})?)', re.IGNORECASE)

# Only fetch from these — everything else is blocked even if DDG returns it
ALLOWED_DOMAINS = {
    "meesho.com", "indiamart.com", "etsy.com", "craftsvilla.com",
    "okhai.com", "jaypore.com", "gaatha.com", "theloom.in",
    "ekscraft.com", "craftshopindia.com", "indianroots.com",
}

# Hard blocklist as a second safety net
BLOCKED_DOMAINS = {
    "amazon.in", "amazon.com", "flipkart.com", "myntra.com",
    "ajio.com", "nykaa.com", "snapdeal.com", "shopclues.com",
    "youtube.com", "instagram.com", "facebook.com", "pinterest.com",
    "udemy.com", "coursera.org", "skillshare.com",  # course sites inflate prices
}

def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "").lower()
    except Exception:
        return ""

def is_allowed_url(url: str) -> bool:
    domain = get_domain(url)
    if any(blocked in domain for blocked in BLOCKED_DOMAINS):
        return False
    if any(allowed in domain for allowed in ALLOWED_DOMAINS):
        return True
    return False  # default deny — only fetch from known-good domains

async def duckduckgo_search(query: str, num_results: int = 10) -> list[str]:
    search_query = (
        f"{query} handmade buy price India "
        f"site:meesho.com OR site:indiamart.com OR site:etsy.com "
        f"OR site:craftsvilla.com OR site:okhai.com OR site:jaypore.com"
    )
    url = "https://html.duckduckgo.com/html/"
    async with httpx.AsyncClient(timeout=15.0, headers=HEADERS, follow_redirects=True) as client:
        resp = await client.post(url, data={"q": search_query})
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for a in soup.select("a.result__url"):
            href = a.get("href", "")
            if not href or not href.startswith("http"):
                continue
            if not is_allowed_url(href):
                continue
            links.append(href)
            if len(links) >= num_results:
                break
        return links

def is_plausible_price(val: float) -> bool:
    if val < 50 or val > 15000:
        return False
    # Round thousands above 5k are usually bulk/wholesale minimums
    if val >= 5000 and val % 1000 == 0:
        return False
    return True

async def scrape_prices_from_url(url: str) -> list[float]:
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=HEADERS, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return []

            # Final domain check after redirects (catches redirect to amazon etc.)
            final_domain = get_domain(str(resp.url))
            if any(blocked in final_domain for blocked in BLOCKED_DOMAINS):
                return []

            soup = BeautifulSoup(resp.text, "html.parser")

            # Target price-specific elements first
            price_els = soup.select(
                "[class*='price'], [class*='Price'], [class*='cost'], "
                "[class*='amount'], [itemprop='price'], [class*='mrp'], "
                "[class*='rate'], [class*='Rate']"
            )
            text = " ".join(el.get_text() for el in price_els)

            # Fall back to page body if nothing found
            if not text.strip():
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator=" ")

            matches = PRICE_RE.findall(text)
            prices = []
            seen = set()
            for m in matches:
                try:
                    val = float(m.replace(",", ""))
                    if val not in seen and is_plausible_price(val):
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

    for url in urls[:6]:
        prices = await scrape_prices_from_url(url)
        if prices:
            all_prices.extend(prices)
            sources.append(url)

    stats = compute_market_stats(all_prices)
    stats["sources"] = sources
    stats["from_cache"] = False

    set_market_cache(cache_key, stats)
    return stats