"""
reindex_only.py

Use this instead of build_market_index.py when the raw fetch + LLM
enrichment are already done and saved (data/market/market_dataset.json
exists) and you only need to rebuild the Chroma embeddings — e.g. after
changing the embedding model, distance space, or embedding_text format.

No Yuukke token needed. No Groq calls. Just re-embeds what's already there.

Run from backend/:
    python -m market_rag.reindex_only
"""

import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from market_rag.market_index_service import build_market_index

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "market" / "market_dataset.json"


def run():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} doesn't exist yet — you need to run "
            "build_market_index.py at least once first."
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # market_dataset.json picked up a handful of duplicate ids across crash/retry
    # runs during enrichment — dedupe here (keep the last occurrence) so Chroma
    # doesn't reject the batch outright.
    deduped = {r["id"]: r for r in dataset}
    dataset = list(deduped.values())

    print(f"Reindexing {len(dataset)} products from cached dataset (no fetch, no LLM calls)...")
    build_market_index(dataset)
    print("Done.")


if __name__ == "__main__":
    run()