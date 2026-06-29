# Price Buddy v2

AI-powered pricing assistant for artisans. Uses Ollama (Llama 3.1 8B) locally + real web scraping for market research.

cd backend
.\venv\Scripts\Activate
uvicorn main:app --reload --port 8080

cd frontend
python -m http.server 3000
## Folder structure

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

## Setup

### 1. Ollama

```bash
# Install Ollama from https://ollama.com
# Then pull the model:
ollama pull llama3.1:8b

# Start Ollama (usually auto-starts, or run manually):
ollama serve
```

### 2. Backend

```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend runs at: http://localhost:8000
API docs at: http://localhost:8000/docs

### 3. Frontend

Just open `frontend/index.html` in your browser — no build step needed.

Or serve it:
```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

## Test data

Try these descriptions:

```
I make hand-painted clay diyas. I spent around ₹120 on materials.
It took me one afternoon to make 10 pieces.

I crochet teddy bears. Each one takes about 5 hours.
Materials cost ₹180 per bear. I sell them for ₹350, is that okay?

I make block-printed cotton tote bags. ₹80 per bag in materials,
takes me about 45 minutes each. Bought supplies 15km away.
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /analyze | Full analysis: AI + market research |
| POST | /market-research | Market research only |
| POST | /chat | Follow-up in existing session |
| GET | /history | List all sessions |
| GET | /history/{id} | Messages for a session |
| GET | /health | Health check |

## Troubleshooting

**"Ollama is not running"** → Run `ollama serve` in a terminal

**"Cannot reach backend"** → Make sure uvicorn is running on port 8000

**Market research returns no data** → DuckDuckGo rate limits; results cached for 6 hrs after first successful fetch

**Slow responses** → Llama 3.1 8B takes 5–20s on CPU. Use GPU if available (Ollama auto-detects).
