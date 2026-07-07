"""
build_market_index.py

End-to-end market-reference pipeline:
  1. Pull every page of Yuukke Marketplace products, save the raw dump.
  2. Enrich each product via market_llm_service (category/materials/keywords).
  3. Save the clean, unified dataset.
  4. Embed it with SentenceTransformer and build a separate ChromaDB
     collection ("market_index"). The existing pricing RAG index is never touched.

Run directly from backend/:
    python -m market_rag.build_market_index
"""

import asyncio

from dotenv import load_dotenv
load_dotenv()  # must run before the market_rag imports below, since they read
                # GROQ_API_KEY / YUUKKE_BEARER_TOKEN from os.environ at import time

from market_rag.yuukke_client import fetch_and_save_all
from market_rag.dataset_builder import build_clean_dataset
from market_rag.market_index_service import build_market_index


async def run():
    print("Step 1/3 — fetching all products from Yuukke Marketplace...")
    raw_products = await fetch_and_save_all()

    print("\nStep 2/3 — enriching products via LLM (category/materials/keywords)...")
    dataset = await build_clean_dataset(raw_products)

    print("\nStep 3/3 — building market_index (FAISS)...")
    build_market_index(dataset)

    print("\nDone. market_index is ready — your pricing RAG index is untouched.")


if __name__ == "__main__":
    asyncio.run(run())