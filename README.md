# Price Buddy v2
# Artisan Buddy

AI-powered pricing assistant for artisans. Uses Ollama (Llama 3.1 8B) locally + real web scraping for market research.
> AI-powered pricing and listing assistant for artisans.

cd backend
.\venv\Scripts\Activate
uvicorn main:app --reload --port 8080
Price Buddy helps artisans estimate fair selling prices using AI reasoning, semantic search, and real marketplace data. It also reviews product photos and provides actionable design feedback before products are listed for sale.

cd frontend
python -m http.server 3000
## Folder structure
---

```
price-buddy/
├── backend/
│   ├── main.py                   # FastAPI app
│   ├── requirements.txt
│   ├── routers/
│   │   ├── analyze.py            # POST /analyze
│   │   ├── market.py             # POST /market-research
│   │   ├── chat.py               # POST /chat
│   │   └── history.py            # GET /history
│   ├── services/
│   │   ├── ollama_service.py     # Llama 3.1 via Ollama
│   │   └── scraper.py            # DuckDuckGo + BeautifulSoup
│   └── db/
│       └── database.py           # SQLite setup + queries
├── frontend/
│   └── index.html                # Single-file frontend
└── README.md
```
## Features

## Setup
### Price Buddy

### 1. Ollama
- AI-assisted price estimation from natural language product descriptions
- Estimates material cost, labor effort, and pricing rationale
- Uses Retrieval-Augmented Generation (RAG) to compare similar marketplace products
- Interactive chat for refining pricing recommendations and follow-up questions

```bash
# Install Ollama from https://ollama.com
# Then pull the model:
ollama pull llama3.1:8b
<p align="center">
  <img src="images/chatprompt.png" width="46%">
  <img src="images/resultprice.png" width="46%">
</p>

# Start Ollama (usually auto-starts, or run manually):
ollama serve
```
---

### 2. Backend
### Design Buddy

```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
- AI-powered product image analysis
- Reviews photography quality and product presentation
- Evaluates craftsmanship, target audience, and listing readiness
- Provides practical suggestions to improve product listings

<p align="center">
  <img src="images/resultdesign.png" width="65%">
</p>

---

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 8307c64 (Update README)
### Custom Marketplace RAG

Instead of relying solely on an LLM's general knowledge, Price Buddy retrieves similar real-world products before generating a recommendation.

The application builds its own marketplace knowledge base by consuming the Yuukke Marketplace API, processing each product into semantic embeddings, and storing them inside ChromaDB for fast similarity search.

During every pricing request, the chatbot retrieves the most relevant marketplace products and supplies them as context to the LLM, producing recommendations that are grounded in actual market data rather than generic estimates.

**Pipeline**

```
Yuukke Marketplace API
        │
        ▼
Download Product Data
        │
        ▼
AI Metadata Extraction
(Category • Materials • Keywords)
        │
        ▼
Sentence Transformer Embeddings
        │
        ▼
ChromaDB Vector Database
        │
        ▼
Semantic Retrieval
        │
        ▼
LLM Price Recommendation
```

<<<<<<< HEAD
---

=======
>>>>>>> db4d45b (readme images, backend market_dataset.json)
=======
Backend runs at: http://localhost:8000
API docs at: http://localhost:8000/docs
---

>>>>>>> 8307c64 (Update README)
## Highlights

- Built a custom Retrieval-Augmented Generation (RAG) pipeline from scratch.
- Created a marketplace knowledge base using **1,400+ real marketplace products**.
- Automatically retrieves product information directly from the Yuukke Marketplace API.
- Enriches marketplace data with AI-generated categories, materials, and searchable keywords.
- Uses Sentence Transformers and ChromaDB for semantic similarity search.
- Combines retrieved marketplace examples with LLM reasoning to generate explainable, market-aware price recommendations.
- Supports multimodal product evaluation through image analysis and design feedback.

---

## Tech Stack

| Layer | Technology |
|--------|------------|
| LLM | Qwen3-32B via Groq |
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
       ChromaDB Marketplace     Design Analysis
            Retrieval                 │
               │                      │
               └──────────┬───────────┘
                          ▼
                   Qwen3-32B Reasoning
                          │
                          ▼
      Price Recommendation + Market Comparison +
             Product Feedback & Explanation
```

### 3. Frontend
---

Just open `frontend/index.html` in your browser — no build step needed.
## Running Locally

Or serve it:
```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```
# Backend
cd backend

python -m venv venv
.\venv\Scripts\Activate.ps1

## Test data
pip install -r requirements.txt

Try these descriptions:
uvicorn main:app --reload --port 8080

# Frontend
cd frontend
python -m http.server 3000
```
I make hand-painted clay diyas. I spent around ₹120 on materials.
It took me one afternoon to make 10 pieces.

I crochet teddy bears. Each one takes about 5 hours.
Materials cost ₹180 per bear. I sell them for ₹350, is that okay?
Backend → `http://localhost:8080`

I make block-printed cotton tote bags. ₹80 per bag in materials,
takes me about 45 minutes each. Bought supplies 15km away.
```
Frontend → `http://localhost:3000`

## API endpoints
API Docs → `http://localhost:8080/docs`

| Method | Path | Description |
|--------|------|-------------|
| POST | /analyze | Full analysis: AI + market research |
| POST | /market-research | Market research only |
| POST | /chat | Follow-up in existing session |
| GET | /history | List all sessions |
| GET | /history/{id} | Messages for a session |
| GET | /health | Health check |
---

## Troubleshooting
## API

**"Ollama is not running"** → Run `ollama serve` in a terminal
| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/analyze` | Complete AI pricing analysis |
| POST | `/market-research` | Marketplace comparison only |
| POST | `/chat` | Continue an existing pricing session |
| POST | `/design-analyze` | Product image analysis |
| POST | `/market-index/build` | Rebuild marketplace knowledge base |
| GET | `/history` | Retrieve previous sessions |
| GET | `/health` | Health check |

**"Cannot reach backend"** → Make sure uvicorn is running on port 8000
---

**Market research returns no data** → DuckDuckGo rate limits; results cached for 6 hrs after first successful fetch
## Future Improvements

**Slow responses** → Llama 3.1 8B takes 5–20s on CPU. Use GPU if available (Ollama auto-detects).
- Support multiple marketplace sources for broader price comparisons.
- Personalized pricing based on artisan experience and region.
- Automatic material detection from uploaded product images.
- Market trend tracking and demand forecasting.
<<<<<<< HEAD
- Export pricing reports for sellers and businesses.
=======
- Export pricing reports for sellers and businesses.
>>>>>>> db4d45b (readme images, backend market_dataset.json)
