# Iryax AI Assistant - Backend API

FastAPI RAG (Retrieval-Augmented Generation) backend service for Iryax AI Assistant. Includes semantic caching, ChromaDB vector store integration, real-time web scraping pipeline, analytics tracking, and streaming NDJSON response generation.

## 🚀 Features

- **RAG Pipeline**: Integrates vector search with ChromaDB and LLM text generation.
- **Streaming Response**: NDJSON token streaming (`/chat`) for real-time frontend responses.
- **Analytics & Health**: Query statistics, latency tracking, and health check endpoints.
- **Feedback Endpoint**: Capture thumbs-up / thumbs-down ratings.
- **Automated Web Scraper**: Scrapes and indexes site contents into vector database.

## 🛠️ Prerequisites

- **Python**: 3.10+
- **Virtual Environment**: `venv`

## 📦 Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ingrainbrands/chatbotbackend.git
   cd chatbotbackend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # Windows
   # source venv/bin/activate # Linux/macOS
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the API Server

### Option 1: Via uvicorn
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: Via Batch Script (Windows)
```powershell
.\run_api.bat
```

Once running, interactive API docs (Swagger UI) are available at [https://apichatbot.iryax.com/docs](https://apichatbot.iryax.com/docs) or [http://localhost:8000/docs](http://localhost:8000/docs).

## 📄 API Endpoints

- `GET /health` — Health check endpoint.
- `POST /chat` — Streamed AI response generation.
- `POST /feedback` — Submit response feedback.
- `GET /analytics` — View system analytics and cache performance.
- `GET /docs` — Swagger UI API documentation.

## 📄 License

Private - Ingrain Brands / Iryax AI
