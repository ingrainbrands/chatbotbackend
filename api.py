import sys
import os
import subprocess
import pathlib
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add current directory to Python's sys.path
ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

try:
    from rag_pipeline import (
        generate_rag_response,
        clear_semantic_cache,
        log_feedback,
        get_analytics_summary
    )
except ImportError:
    from backend.rag_pipeline import (
        generate_rag_response,
        clear_semantic_cache,
        log_feedback,
        get_analytics_summary
    )

SCRAPER_LOG_PATH = ROOT / "scraper.log"
scraper_process = None

app = FastAPI(
    title="Iryax AI Assistant API",
    description="API for Iryax RAG Assistant",
    version="1.0.0"
)

@app.on_event("startup")
async def start_background_scraper():
    """Automatically launch the background web scraper when backend server starts."""
    global scraper_process
    scraper_path = ROOT / "scraper.py"
    if not scraper_path.exists():
        scraper_path = ROOT / "backend" / "scraper.py"

    if scraper_path.exists():
        scraper_process = subprocess.Popen(
            [sys.executable, str(scraper_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )

@app.on_event("shutdown")
async def stop_background_scraper():
    """Cleanly terminate the background web scraper when backend shuts down."""
    global scraper_process
    if scraper_process:
        try:
            scraper_process.terminate()
        except Exception:
            pass
        scraper_process = None


# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",   # Vite dev server (current)
        "http://localhost:5175",   # Vite fallback port
        "http://localhost:3000",   # CRA / other dev servers
        "https://chatbot.iryax.com",
        "https://iryax.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    # Accept both 'message' and 'question' (frontend sends 'question')
    message: str = ""
    question: str = ""
    history: List[Dict[str, str]] = []
    website_id: str = "https://iryax.com"
    session_id: str = "default_session"

    @property
    def user_query(self) -> str:
        """Return whichever field the client sent."""
        return (self.question or self.message).strip()

class FeedbackRequest(BaseModel):
    query: str
    rating: str   # 'thumbs_up' or 'thumbs_down'
    comment: str = ""

# ── Health check ───────────────────────────────────────────────────────────────
@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    """Verify the API is running."""
    return {"status": "ok", "message": "Iryax AI Assistant API is running"}

# ── Chat — POST (streaming NDJSON) ─────────────────────────────────────────────
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Accepts {question, message, history, session_id, website_id}.
    Returns streaming NDJSON: one JSON object per line.
    Final line includes: {answer, sources, session_id}
    """
    query = req.user_query
    if not query:
        return {"error": "Empty message received."}

    def generator():
        for chunk in generate_rag_response(query, req.history):
            yield chunk

    return StreamingResponse(generator(), media_type="application/x-ndjson")

# ── Chat — GET (friendly info for browsers/tools using wrong method) ───────────
@app.get("/chat")
async def chat_get_info():
    """Returns a helpful message when /chat is hit with GET instead of POST."""
    return {
        "endpoint": "/chat",
        "method_required": "POST",
        "body": {"message": "string", "history": "list"},
        "docs": "https://apichatbot.iryax.com/docs",
    }

# ── /doc  →  /docs redirect ────────────────────────────────────────────────────
@app.get("/doc", include_in_schema=False)
async def doc_redirect():
    """/doc → /docs (FastAPI interactive Swagger UI)."""
    return RedirectResponse(url="/docs")

# ── Favicon — suppress browser 404 noise ──────────────────────────────────────
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve a minimal purple SVG favicon so browsers stop logging 404s."""
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        b'<rect width="32" height="32" rx="8" fill="#7c3aed"/>'
        b'<text x="16" y="22" text-anchor="middle" font-size="18" '
        b'font-family="Arial" fill="white">&#x2726;</text>'
        b'</svg>'
    )
    return Response(content=svg, media_type="image/svg+xml")

# ── Feedback — POST ────────────────────────────────────────────────────────────
@app.post("/feedback")
async def post_feedback(req: FeedbackRequest):
    """Log user feedback (thumbs up / thumbs down with optional comment)."""
    return log_feedback(req.query, req.rating, req.comment)

# ── Analytics — GET ────────────────────────────────────────────────────────────
@app.get("/analytics")
async def get_analytics():
    """Retrieve query statistics, routing distribution, and cache hit rates."""
    return get_analytics_summary()

# ── Cache Clear — POST ──────────────────────────────────────────────────────────
@app.post("/cache/clear")
async def clear_cache():
    """Invalidate in-memory response cache."""
    cleared = clear_semantic_cache()
    return {"status": "success", "cleared_entries": cleared}

# ── Scraper Log — GET ────────────────────────────────────────────────────────────
@app.get("/scraper/log")
async def get_scraper_log(lines: int = 50):
    """Retrieve the recent history of the scraper from scraper.log."""
    if not SCRAPER_LOG_PATH.exists():
        return {"status": "error", "message": "Scraper log not found"}
    try:
        with open(SCRAPER_LOG_PATH, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            return {"status": "success", "log": "".join(all_lines[-lines:])}
    except Exception as e:
        return {"status": "error", "message": str(e)}

