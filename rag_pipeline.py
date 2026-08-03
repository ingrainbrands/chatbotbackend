import json
import math
import pathlib
# pyrefly: ignore [missing-import]
import ollama
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

load_dotenv()
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "info@ingrainsystem.com")
CONTACT_PHONE = os.getenv("CONTACT_PHONE", "+91-9010481048")

# Config
CHUNKS_PATH = pathlib.Path(__file__).parent / "data" / "processed" / "iryax_chunks.json"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DB_DIR = pathlib.Path(__file__).parent / "data" / "chromadb"

# Initialize ChromaDB and Embedding Function
chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
embed_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="iryax_context_v2", embedding_function=embed_fn)

def load_data_if_empty():
    """Load chunks into ChromaDB if the collection is empty."""
    if collection.count() > 0:
        return
    
    if not CHUNKS_PATH.exists():
        print("No chunks found at", CHUNKS_PATH)
        return
        
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    ids = []
    documents = []
    metadatas = []
    
    for c in chunks:
        ids.append(c["chunk_id"])
        documents.append(c["content"])
        metadatas.append({
            "url": c.get("url", ""),
            "label": c.get("label", "")
        })
        
    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Loaded {len(chunks)} chunks into ChromaDB.")
    else:
        print("No chunks found in iryax_chunks.json to load into ChromaDB.")

# Load immediately upon import
load_data_if_empty()

import time
import threading

def _prewarm():
    try:
        ollama.generate(model=MODEL_NAME, prompt="", keep_alive=-1)
    except Exception:
        pass

threading.Thread(target=_prewarm, daemon=True).start()

import re

CACHE_FILE = pathlib.Path(__file__).parent / "data" / "processed" / "response_cache.json"

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

RESPONSE_CACHE = _load_cache()
CACHE_EMBEDDINGS = {}

def _update_cache_embeddings():
    try:
        for k in list(RESPONSE_CACHE.keys()):
            if k not in CACHE_EMBEDDINGS:
                CACHE_EMBEDDINGS[k] = embed_fn([k])[0]
    except Exception:
        pass

threading.Thread(target=_update_cache_embeddings, daemon=True).start()

def _cosine_similarity(vec1, vec2):
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def find_semantic_cache_match(query: str, threshold: float = 0.82) -> str:
    """Check if query is semantically similar (>= threshold) to any cached question."""
    if not RESPONSE_CACHE:
        return None
    try:
        query_vec = embed_fn([query])[0]
        best_score = -1.0
        best_key = None
        for k, vec in CACHE_EMBEDDINGS.items():
            score = _cosine_similarity(query_vec, vec)
            if score > best_score:
                best_score = score
                best_key = k
        if best_score >= threshold and best_key in RESPONSE_CACHE:
            return RESPONSE_CACHE[best_key]
    except Exception:
        pass
    return None

def _save_cache():
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(RESPONSE_CACHE, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _update_single_embedding(key: str):
    try:
        CACHE_EMBEDDINGS[key] = embed_fn([key])[0]
    except Exception:
        pass

def get_cache_key(message: str) -> str:
    cleaned = re.sub(r'[^\w\s]', '', message.strip().lower())
    return ' '.join(cleaned.split())


# Load structured intents database
INTENTS_PATH = pathlib.Path(__file__).parent / "data" / "processed" / "structured_intents.json"
STRUCTURED_INTENTS = []
LAST_INTENT_LOAD_TIME = 0.0

PAGES_FILE = pathlib.Path(__file__).parent / "data" / "raw" / "iryax_pages.json"
RAW_PAGES_CACHE = {}
LAST_PAGES_LOAD_TIME = 0.0

def load_pages_cache():
    global RAW_PAGES_CACHE, LAST_PAGES_LOAD_TIME
    if PAGES_FILE.exists():
        try:
            mtime = os.path.getmtime(PAGES_FILE)
            if mtime > LAST_PAGES_LOAD_TIME:
                with open(PAGES_FILE, "r", encoding="utf-8") as f:
                    RAW_PAGES_CACHE = json.load(f)
                LAST_PAGES_LOAD_TIME = mtime
        except Exception as e:
            print("Error loading pages cache:", e)

INTENT_EMBEDDINGS = {}

def _update_intent_embeddings():
    try:
        for intent in STRUCTURED_INTENTS:
            texts_to_embed = intent.get("exact_queries", []) + intent.get("phrases", [])
            for text in texts_to_embed:
                if text not in INTENT_EMBEDDINGS:
                    INTENT_EMBEDDINGS[text] = embed_fn([text])[0]
    except Exception as e:
        print("EMBEDDING UPDATE ERROR:", e)

def load_structured_intents():
    global STRUCTURED_INTENTS, LAST_INTENT_LOAD_TIME
    if INTENTS_PATH.exists():
        try:
            mtime = os.path.getmtime(INTENTS_PATH)
            if mtime > LAST_INTENT_LOAD_TIME:
                with open(INTENTS_PATH, "r", encoding="utf-8") as f:
                    STRUCTURED_INTENTS = json.load(f).get("intents", [])
                LAST_INTENT_LOAD_TIME = mtime
                # INTENT_EMBEDDINGS will be lazily loaded in main thread
        except Exception as e:
            print("Error loading structured intents:", e)

# Initial load
load_structured_intents()
load_pages_cache()

def check_dynamic_intent(message: str):
    """
    Structured Database / Instant Answer lookup for predictable Dynamic Intents,
    loaded dynamically from structured_intents.json to avoid hardcoded text in Python.
    """
    msg_lower = message.strip().lower()
    


    load_structured_intents()

    msg_lower = message.strip().lower()
    clean_msg = re.sub(r'[^\w\s]', '', msg_lower)
    words = clean_msg.split()

    # 1. Check exact query match for all intents
    for intent in STRUCTURED_INTENTS:
        if clean_msg in intent.get("exact_queries", []):
            return (intent["name"], intent["answer"], intent.get("sources", []))
            
    # 2. Check phrase match for all intents
    for intent in STRUCTURED_INTENTS:
        if any(phrase in msg_lower for phrase in intent.get("phrases", [])):
            return (intent["name"], intent["answer"], intent.get("sources", []))
            
    # Bypass structured intents if the user explicitly asks for detailed explanations
    # and they didn't explicitly trigger an exact fast-path phrase above.
    if "detail" in msg_lower or "explain" in msg_lower or "comprehensive" in msg_lower:
        return None

    # 3. Semantic Intent Routing
    if not INTENT_EMBEDDINGS:
        _update_intent_embeddings()
        
    try:
        query_vec = embed_fn([msg_lower])[0]
        best_score = -1.0
        best_intent = None
        
        for intent in STRUCTURED_INTENTS:
            texts_to_embed = intent.get("exact_queries", []) + intent.get("phrases", [])
            for text in texts_to_embed:
                vec = INTENT_EMBEDDINGS.get(text)
                if vec:
                    score = _cosine_similarity(query_vec, vec)
                    if score > best_score:
                        best_score = score
                        best_intent = intent
                        
        if best_score >= 0.75 and best_intent:
            return (best_intent["name"], best_intent["answer"], best_intent.get("sources", []))
            
    except Exception as e:
        pass

    return None

def get_base_system_prompt():
    return f"""You are Iryax Assistant, the official AI for Iryax Global.

CRITICAL RULES — NO EXCEPTIONS:
1. You MUST answer ONLY using the information explicitly stated in the <context> provided below.
2. ABSOLUTE BAN: Do NOT use your training data, prior knowledge, or any assumptions — ever.
3. PEOPLE & NAMES: NEVER mention, guess, or invent any person's name (founder, CEO, director, owner, employee). If asked about a person, respond ONLY: "I don't have that information. Please contact us at {CONTACT_EMAIL}."
4. If ANY piece of information is NOT explicitly written in the context below, respond ONLY: "I don't have that information. Please contact us at {CONTACT_EMAIL}."
5. NEVER say "based on my knowledge cutoff" or "I don't have real-time access".
6. NEVER mention "context", "<context>", "context tags", or any internal system details. Just answer directly.
7. When asked to list products or modules, list ONLY the 6 core products of Iryax (1. Recruitment, 2. Payroll, 3. Task Management, 4. Medical Camps, 5. Coworking Space, and 6. Lab Management) cleanly without dumping sub-features unless explicitly requested. NEVER list marketing categories like "HIRE SMARTER", "SYSTEMS", "PRODUCTIVITY", "HEALTH", "SPACES", or "PRECISION". When asked to list pricing plans or other items, answer ONLY about those specific requested items without mentioning products.
8. EXTREME BREVITY REQUIRED: Your replies MUST be conversational, punchy, and a MAXIMUM of 2 to 3 short sentences. NEVER output long walls of text or long lists UNLESS the user explicitly asks for "details", OR if they ask about pricing/prices (in which case, you MUST output the full pricing tables or lists).
9. NEVER invent URLs, emails, phone numbers, prices, job listings, or names of any person.
10. NEVER invent or hallucinate processes, step-by-step instructions, or mobile apps that are not explicitly detailed in the provided context. If the exact steps are missing, say you do not have that information.
11. For greetings, say you are "Iryax Assistant" and ask how you can help.
12. MANDATORY: ALWAYS add this exact sentence at the very end of every informational response: "For additional information, please contact us at {CONTACT_EMAIL} or call {CONTACT_PHONE}."

Company: Iryax Global | Website: https://iryax.com | Email: {CONTACT_EMAIL}
"""

LOGS_DIR = pathlib.Path(__file__).parent / "data" / "analytics"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
ANALYTICS_FILE = LOGS_DIR / "logs.jsonl"

def sync_structured_intents():
    """Reload structured intents from disk and sync with latest scraped web pages if updated."""
    global STRUCTURED_INTENTS
    if INTENTS_PATH.exists():
        try:
            with open(INTENTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                STRUCTURED_INTENTS = data.get("intents", [])
                
            # Live intent sync logic removed. We rely on RAG for dynamic website data.
        except Exception as e:
            print("Error syncing structured intents:", e)
            
# Force reload

# Force reload 3
def clear_semantic_cache() -> int:
    """Clear in-memory and disk response cache (called on delta sync update)."""
    count = len(RESPONSE_CACHE)
    RESPONSE_CACHE.clear()
    CACHE_EMBEDDINGS.clear()
    if CACHE_FILE.exists():
        try:
            CACHE_FILE.unlink()
        except Exception:
            pass
    # Auto-sync structured intents database when web scraper runs
    sync_structured_intents()
    return count

def log_conversation(message: str, response: str, sources: list, route: str, latency_ms: float):
    def _write():
        try:
            entry = {
                "timestamp": time.time(),
                "query": message,
                "response": response,
                "sources": sources,
                "route": route,
                "latency_ms": round(latency_ms, 2)
            }
            with open(ANALYTICS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
    threading.Thread(target=_write, daemon=True).start()

def log_feedback(query: str, rating: str, comment: str = "") -> dict:
    try:
        entry = {
            "timestamp": time.time(),
            "type": "feedback",
            "query": query,
            "rating": rating,
            "comment": comment
        }
        with open(ANALYTICS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_analytics_summary() -> dict:
    if not ANALYTICS_FILE.exists():
        return {"total_queries": 0, "cache_hits": 0, "routes": {}, "avg_latency_ms": 0, "feedback": {"thumbs_up": 0, "thumbs_down": 0}}
    
    total = 0
    cache_hits = 0
    routes = {}
    total_lat = 0.0
    feedback_counts = {"thumbs_up": 0, "thumbs_down": 0}
    
    with open(ANALYTICS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("type") == "feedback":
                    r = data.get("rating")
                    if r in feedback_counts:
                        feedback_counts[r] += 1
                else:
                    total += 1
                    route = data.get("route", "rag_llm")
                    routes[route] = routes.get(route, 0) + 1
                    if route == "cache":
                        cache_hits += 1
                    total_lat += data.get("latency_ms", 0.0)
            except Exception:
                continue
                
    return {
        "total_queries": total,
        "cache_hits": cache_hits,
        "routes": routes,
        "avg_latency_ms": round(total_lat / max(1, total), 2),
        "feedback": feedback_counts
    }

CONTACT_FOOTER = f"\n\nFor additional information, please contact us at {CONTACT_EMAIL} or call {CONTACT_PHONE}."

# ── Broad set of Iryax-related terms for fast off-topic detection ─────────────
# If the user message contains NONE of these words, skip ALL LLM/embedding work
# and return the FAQ list instantly.
IRYAX_KEYWORDS = {
    # company
    "iryax", "ingrain", "ingrainsystem",
    # products
    "product", "products", "module", "modules", "feature", "features",
    "recruitment", "recruit", "hire", "hiring", "applicant", "candidate", "job", "jobs",
    "payroll", "attendance", "leave", "salary", "timesheet", "roster", "shift",
    "task", "tasks", "project", "sprint", "agile", "collaboration",
    "medical", "camp", "camps", "health", "patient", "volunteer",
    "workspace", "coworking", "cabin", "desk", "room", "booking", "iot",
    "lab", "laboratory",
    # pricing
    "price", "pricing", "cost", "fee", "plan", "plans", "subscription",
    "basic", "premium", "enterprise", "custom",
    # company info
    "about", "company", "overview", "who", "what", "how", "explain", "tell",
    "works", "work", "secure", "security", "privacy", "data",
    # contact / support
    "contact", "email", "phone", "support", "address", "office", "location",
    "demo", "book", "schedule", "whatsapp",
    # careers
    "career", "careers", "vacancy", "vacancies", "opening", "internship", "position",
    # greetings already handled by intents but keep here as safe fallback
    "hi", "hello", "hey",
}

# ── FAQ shown instantly when query is off-topic (no LLM/embedding used) ───────
FAQ_RESPONSE = """I can only help with questions about **Iryax Global**. Here are some things I can answer:

❓ **Products & Features**
- What products does Iryax offer?
- Tell me about Recruitment / Payroll / Task Management / Workspace / Medical Camps

💰 **Pricing**
- What are the pricing plans?
- Basic / Premium / Enterprise / Custom plan details

🏢 **Company**
- About Iryax Global
- How does Iryax work?
- Is my data secure with Iryax?

📞 **Contact & Demo**
- How can I contact Iryax support?
- How do I book a demo?

💼 **Careers**
- What job openings are available at Iryax?

For anything else not listed here, please contact us at {CONTACT_EMAIL} or call {CONTACT_PHONE}.""".format(CONTACT_EMAIL=CONTACT_EMAIL, CONTACT_PHONE=CONTACT_PHONE)

def _is_off_topic(message: str) -> bool:
    """Return True if message contains NO Iryax-related keywords at all.
    This check runs before any embedding/ChromaDB/LLM work, making it free."""
    words = set(re.sub(r'[^\w\s]', '', message.lower()).split())
    return not words.intersection(IRYAX_KEYWORDS)

def _ensure_contact_footer(text: str) -> str:
    """Guarantee every response ends with the standard contact footer."""
    if "For additional information" not in text:
        return text.rstrip() + CONTACT_FOOTER
    return text


# ── Topics we know are NOT in our scraped data — must never reach the LLM ─────
# The 3B model will hallucinate names/facts for these if given the chance.
_NO_INFO_PATTERNS = re.compile(
    r"\b(founder|co-founder|cofounder|ceo|chief executive|chairman|director|owner|"
    r"president|managing director|md\b|founded by|started by|created by|built by|"
    r"established by|who (started|created|built|runs|owns|leads|founded) iryax|"
    r"who (is|are) the (founder|ceo|owner|head|team|leader|management))\b",
    re.IGNORECASE
)

_NO_INFO_RESPONSE = (
    f"I don't have information about the founders or leadership team of Iryax Global. "
    f"For additional information, please contact us at {CONTACT_EMAIL} or call {CONTACT_PHONE}."
)

def generate_rag_response(user_message: str, history: list):
    start_t = time.time()

    # ── Off-topic guard: fires before ANY embedding/ChromaDB/LLM work ─────────
    # If the message has zero Iryax-related words, skip everything and show FAQ.
    if _is_off_topic(user_message):
        log_conversation(user_message, FAQ_RESPONSE, [], "off_topic_faq", (time.time() - start_t) * 1000)
        for word in FAQ_RESPONSE.split(" "):
            yield json.dumps({"token": word + " ", "sources": []}) + "\n"
        return

    # ── Unanswerable guard: topics not in our data — stop before LLM ──────────
    # Prevents the model from hallucinating names, founders, CEOs, etc.
    if _NO_INFO_PATTERNS.search(user_message):
        log_conversation(user_message, _NO_INFO_RESPONSE, [], "no_info_guard", (time.time() - start_t) * 1000)
        yield json.dumps({"token": _NO_INFO_RESPONSE, "sources": []}) + "\n"
        return

    # ── Query Router: Dynamic Intent (PRICING / JOBS / CONTACT / PRODUCTS) ───
    dynamic_match = check_dynamic_intent(user_message)
    if dynamic_match:
        intent_name, instant_ans, intent_sources = dynamic_match
        instant_ans = _ensure_contact_footer(instant_ans)
        log_conversation(user_message, instant_ans, intent_sources, f"instant_{intent_name}", (time.time() - start_t) * 1000)
        words = instant_ans.replace("\r", "").split(" ")
        for w in words:
            yield json.dumps({"token": w + " ", "sources": intent_sources}) + "\n"
        return

    # ── Query Router: Cache Check (Lexical Exact Match) ──────────────────────
    cache_key = get_cache_key(user_message)
    if cache_key in RESPONSE_CACHE:
        cached_ans = _ensure_contact_footer(RESPONSE_CACHE[cache_key])
        log_conversation(user_message, cached_ans, ["Cache"], "cache", (time.time() - start_t) * 1000)
        yield json.dumps({"token": cached_ans, "sources": ["Cache"]}) + "\n"
        return

    # ── Query Router: Cache Check (Semantic Match >= 82%) ────────────────────
    semantic_ans = find_semantic_cache_match(user_message, threshold=0.82)
    if semantic_ans:
        semantic_ans = _ensure_contact_footer(semantic_ans)
        RESPONSE_CACHE[cache_key] = semantic_ans
        threading.Thread(target=_update_single_embedding, args=(cache_key,), daemon=True).start()
        threading.Thread(target=_save_cache, daemon=True).start()
        log_conversation(user_message, semantic_ans, ["Cache (Semantic)"], "semantic_cache", (time.time() - start_t) * 1000)
        yield json.dumps({"token": semantic_ans, "sources": ["Cache (Semantic)"]}) + "\n"
        return

    # ── Keyword-based page injection ─────────────────────────────────────────
    load_pages_cache()
    pinned_docs: list[tuple[str, str]] = []   # (url, markdown)

    if RAW_PAGES_CACHE:
        pages = RAW_PAGES_CACHE

        msg_lower = user_message.lower()

        career_kw  = {"job", "jobs", "career", "careers", "opening", "openings",
                      "vacancy", "vacancies", "apply", "hiring", "recruit", "position"}
        pricing_kw = {"price", "pricing", "plan", "plans", "cost", "fee", "subscription",
                      "basic", "premium", "enterprise", "custom plan"}
        contact_kw = {"contact", "email", "phone", "address", "office", "location", "demo"}
        product_kw = {"product", "products", "feature", "features", "module", "modules",
                      "what do you offer", "what does iryax"}
        about_kw   = {"about iryax", "who is iryax", "what is iryax", "tell me about iryax",
                      "company overview", "who are iryax", "tell me about you",
                      "info about iryax", "what does iryax do", "about company", "iryax overview",
                      "founder", "co-founder", "ceo", "owner", "leadership", "team",
                      "founded", "established", "started by", "created by", "built by"}
        
        workspace_kw = {"coworking", "workspace", "cabin", "desk", "room", "price","pricing","seat","seats","meeting room"}
        recruitment_kw = {"recruitment", "hiring software", "ats", "applicant tracking"}
        payroll_kw = {"payroll", "attendance", "leave management", "timesheet"}
        task_kw = {"task management", "sprint", "agile"}
        camp_kw = {"medical camp", "health camp"}
        lab_kw = {"lab management", "laboratory"}

        def matches(keywords):
            return any(k in msg_lower for k in keywords)

        def get_trimmed(text, limit=1800):
            # Strip markdown images to avoid wasting character budget on image URLs
            clean = re.sub(r'!\[.*?\]\(.*?\)', '', text)
            # Strip repetitive navigation buttons like [Book Demo](/contact)
            clean = re.sub(r'\[(?:Book Demo|Get Started|Learn More|About|View|Login|Client Login)\]\(.*?\)', '', clean, flags=re.IGNORECASE)
            # Collapse multiple blank lines
            clean = re.sub(r'\n{3,}', '\n\n', clean).strip()
            return clean[:limit] + "..." if len(clean) > limit else clean

        if matches(about_kw):
            for url in ("https://iryax.com/about", "https://iryax.com"):
                if url in pages:
                    pinned_docs.append((url, get_trimmed(pages[url]["markdown"])))
        if matches(career_kw):
            for url in ("https://iryax.com/careers",):
                if url in pages:
                    pinned_docs.append((url, get_trimmed(pages[url]["markdown"])))
        if matches(pricing_kw):
            for url in ("https://iryax.com/price",):
                if url in pages:
                    pinned_docs.append((url, get_trimmed(pages[url]["markdown"])))
        if matches(contact_kw):
            for url in ("https://iryax.com/contact",):
                if url in pages:
                    pinned_docs.append((url, get_trimmed(pages[url]["markdown"])))
        if matches(product_kw):
            for url in ("https://iryax.com/products",):
                if url in pages:
                    pinned_docs.append((url, get_trimmed(pages[url]["markdown"])))
        if matches(workspace_kw):
            for url in ("https://iryax.com/workspace",):
                if url in pages:
                    pinned_docs.append((url, get_trimmed(pages[url]["markdown"])))
        if matches(recruitment_kw):
            for url in ("https://iryax.com/recruitment",):
                if url in pages:
                    pinned_docs.append((url, get_trimmed(pages[url]["markdown"])))
        if matches(payroll_kw):
            for url in ("https://iryax.com/attendance",):
                if url in pages:
                    pinned_docs.append((url, get_trimmed(pages[url]["markdown"])))
        if matches(task_kw):
            for url in ("https://iryax.com/task-management",):
                if url in pages:
                    pinned_docs.append((url, get_trimmed(pages[url]["markdown"])))
        if matches(camp_kw):
            for url in ("https://iryax.com/camp",):
                if url in pages:
                    pinned_docs.append((url, get_trimmed(pages[url]["markdown"])))

    # ── Semantic retrieval from ChromaDB ──────────────────────────────────────
    results = collection.query(
        query_texts=[user_message],
        n_results=2
    )

    context_parts = []
    sources = []

    # Pinned pages go first (highest priority)
    pinned_urls = set()
    for url, markdown in pinned_docs:
        context_parts.append(f"[Source: Iryax Global — {url}]\n{markdown}")
        sources.append(url)   # plain URL
        pinned_urls.add(url)

    # Then semantic results (skip if already pinned)
    if results["documents"] and results["documents"][0]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            if meta["url"] not in pinned_urls:
                trimmed_doc = get_trimmed(doc, limit=600)
                context_parts.append(f"[Source: {meta['label']} — {meta['url']}]\n{trimmed_doc}")
                sources.append(meta["url"])   # plain URL

    context_str = "\n\n---\n\n".join(context_parts)

    base_prompt = get_base_system_prompt()

    if context_str:
        system_content = (
            base_prompt
            + "\n\n<context>\n"
            + context_str
            + "\n</context>"
            + f"\n\nREMINDER: You MUST append the contact details at the end of your response: 'For additional information, please contact us at {CONTACT_EMAIL} or call {CONTACT_PHONE}.'"
        )
    else:
        system_content = base_prompt + "\n\nNote: No relevant Iryax data found. Say you don't have that information."
        
    messages = [{"role": "system", "content": system_content}]
    messages.extend(history[-2:])
    
    enforced_user_message = user_message + f'\n\n(Remember to append the contact details: "For additional information, please contact us at {CONTACT_EMAIL} or call {CONTACT_PHONE}.")'
    messages.append({"role": "user", "content": enforced_user_message})
    
    try:
        stream = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            keep_alive=-1,
            options={
                # For interactive chat we limit output length to keep latency reasonable.
                "num_thread": os.cpu_count() or 8,
                "num_ctx": 2048,   # larger context so full pricing/career pages fit
                "num_predict": 600, # enough to complete all 4 pricing plans
                "temperature": 0.1,
            },
        )
        # Yield a heartbeat comment so the HTTP read timeout doesn't fire
        # while Ollama is warming up / generating the first token.
        
        full_response = ""
        for chunk_token in stream:
            token = chunk_token["message"]["content"]
            full_response += token
            yield json.dumps({"token": token, "sources": sources}) + "\n"
            
        RESPONSE_CACHE[cache_key] = full_response
        threading.Thread(target=_update_single_embedding, args=(cache_key,), daemon=True).start()
        threading.Thread(target=_save_cache, daemon=True).start()
        log_conversation(user_message, full_response, sources, "rag_llm", (time.time() - start_t) * 1000)
    except Exception as e:
        yield json.dumps({"token": f"\\n\\n**Error:** {e}", "sources": []}) + "\n"

# Backend forced reload to clear in-memory cache again!
