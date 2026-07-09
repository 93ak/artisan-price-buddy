"""
migrate_market_data.py

One-off: load your existing (already LLM-enriched) data/market/market_dataset.json
straight into Postgres. No scraping, no LLM calls — just re-embeds and inserts.

Run locally, pointed at your Render Postgres, from backend/:
    python -m market_rag.migrate_market_data

Needs DATABASE_URL set to your Render Postgres's *External* connection string
(available on the database's page in the Render dashboard) when running from
your own machine.
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
            f"{DATASET_PATH} not found — this migrates your existing enriched "
            f"dataset, it doesn't build one from scratch. If you don't have "
            f"this file, run build_market_index.py's full pipeline instead."
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} already-enriched products from {DATASET_PATH}")
    print("Embedding + inserting into Postgres market_index table...")
    build_market_index(dataset)
    print("Done.")


if __name__ == "__main__":
    run()