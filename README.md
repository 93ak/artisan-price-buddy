# Artisan Buddy

> AI-powered pricing, listing, and market insight assistant for artisans.

Artisan Buddy helps artisans estimate fair selling prices using AI reasoning, semantic search, and real marketplace data. It reviews product photos and provides actionable design feedback, surfaces market and opportunity insights for new product ideas, and connects the whole workflow together before products are listed for sale.

<p align="center">
  <img src="images/homepage.png" width="46%">
</p>

---

## Features

### Price Buddy

- AI-assisted price estimation from natural language product descriptions
- Estimates material cost, labor effort, and pricing rationale
- Uses Retrieval-Augmented Generation (RAG) to compare similar marketplace products
- Interactive chat for refining pricing recommendations and follow-up questions

<p align="center">
  <img src="images/pricebuddy.png" width="46%">
</p>

<p align="center">
  <img src="images/pricebreakdown.png" width="46%">
  <img src="images/rating.png" width="46%">
</p>

---

### Design Buddy

- AI-powered product image analysis
- Reviews photography quality and product presentation
- Evaluates craftsmanship, target audience, and listing readiness
- Provides practical suggestions to improve product listings

<p align="center">
  <img src="images/resultdesign.png" width="65%">
</p>

---

### Business Buddy

- Market and opportunity insights for a product idea — from a typed description or a photo
- Demand and competition snapshot, best-selling seasons, and crafting difficulty at a glance
- Target audience and buying motivation breakdown
- Similar products, business opportunities, risks, and marketing ideas, expandable on demand
- Photo upload path uses the vision model to identify the product before running the analysis

<p align="center">
  <img src="images/business.png" width="65%">
</p>

---

### Connect

<p align="center">
  <img src="images/connect.png" width="65%">
</p>

<!--
  Note: add a line or two here describing what "Connect" does — I don't have
  details on this feature beyond the screenshot filename, so I've left it as
  a placeholder. Happy to write the description once you tell me what it covers.
-->

---

## Highlights

- Built a custom Retrieval-Augmented Generation (RAG) pipeline from scratch.
- Created a marketplace knowledge base using **1,400+ real marketplace products**.
- Automatically retrieves product information directly from the Marketplace API.
- Enriches marketplace data with AI-generated categories, materials, and searchable keywords.
- Uses Sentence Transformers and ChromaDB for semantic similarity search.
- Combines retrieved marketplace examples with LLM reasoning to generate explainable, market-aware price recommendations.
- Supports multimodal product evaluation through image analysis and design feedback.
- Business Buddy extends the same vision pipeline to turn a product photo directly into market insights.

---

## Tech Stack

| Layer | Technology |
|--------|------------|
| LLM | Qwen3-27B via Groq |
| Vision Model | Qwen3.6-27B via Groq |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB |
| Backend | FastAPI |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |

---

## System Architecture

```text
                    User Input
               ┌─────────┴─────────┐
               ▼                   ▼
      Product Description     Product Image
               │                   │
               ▼                   ▼
     Sentence Transformers     Vision Model
               │                   │
               ▼                   ▼
       ChromaDB Marketplace     Design Analysis /
            Retrieval          Business Insights
               │                      │
               └──────────┬───────────┘
                          ▼
                   Qwen3-27B Reasoning
                          │
                          ▼
      Price Recommendation + Market Comparison +
        Product Feedback + Business Insights
```

---

## Running Locally

```bash
# Backend
cd backend

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

uvicorn main:app --reload --port 8080

# Frontend
cd frontend
python -m http.server 3000
```

Backend → `http://localhost:8080`

Frontend → `http://localhost:3000`

API Docs → `http://localhost:8080/docs`

---

## API

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/analyze` | Complete AI pricing analysis |
| POST | `/market-research` | Marketplace comparison only |
| POST | `/chat` | Continue an existing pricing session |
| POST | `/design-analyze` | Product image analysis |
| POST | `/business-buddy/analyze` | Market and opportunity insights from a product description |
| POST | `/business-buddy/analyze-image` | Market and opportunity insights from a product photo |
| POST | `/market-index/build` | Rebuild marketplace knowledge base |
| GET | `/history` | Retrieve previous sessions |
| GET | `/health` | Health check |

---

## Future Improvements

- Support multiple marketplace sources for broader price comparisons.
- Personalized pricing based on artisan experience and region.
- Automatic material detection from uploaded product images.
- Market trend tracking and demand forecasting.
- Export pricing reports for sellers and businesses.