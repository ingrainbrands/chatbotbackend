from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict

# Import the RAG pipeline
from .rag_pipeline import generate_rag_response

app = FastAPI(title="Iryax AI Backend")

class ChatRequest(BaseModel):
    user_message: str
    history: List[Dict[str, str]] = []

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Accepts a user message and conversation history.
    Streams the response back to the client using Server-Sent Events (SSE) or simple line-by-line JSON.
    """
    generator = generate_rag_response(request.user_message, request.history)
    return StreamingResponse(generator, media_type="application/x-ndjson")

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
