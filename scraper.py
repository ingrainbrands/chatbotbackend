"""
Iryax Website Scraper
=====================
Pipeline (runs every 1 hour):
  Playwright (render JS)
    → BeautifulSoup (clean HTML)
    → Convert to Markdown
    → Compare current vs previous (page-level)
    → If changed: re-chunk only changed pages, update ChromaDB
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import json
import time
import hashlib
import pathlib
import schedule
import logging
from urllib.parse import urldefrag

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import chromadb
from chromadb.utils import embedding_functions

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT         = pathlib.Path(__file__).parent
PAGES_FILE   = ROOT / "data" / "raw"       / "iryax_pages.json"     # per-page markdown cache
OUTPUT_FILE  = ROOT / "data" / "processed" / "iryax_chunks.json"    # all chunks
DB_DIR       = ROOT / "data" / "chromadb"
LOG_FILE     = ROOT / "scraper.log"

class UnbufferedFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        UnbufferedFileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs from httpx/httpcore (used by Ollama) in scraper.log
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def clear_log_file():
    """Clear scraper.log every 3 hours to keep log size manageable."""
    try:
        for handler in logging.root.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")
        logger.info("[Log Cleared] scraper.log has been cleared (scheduled every 3 hours).")
    except Exception as e:
        logger.warning(f"Failed to clear log file: {e}")


TARGET_URL   = "https://iryax.com"
CRAWL_LIMIT  = 200


# ── HTML → Clean Markdown ─────────────────────────────────────────────────────
def html_to_markdown(html: str) -> str:
    """Use BeautifulSoup to strip noise, then convert remaining HTML to markdown."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove tags that add no content value
    for tag in soup(["script", "style", "noscript", "iframe",
                     "header", "footer", "nav", "aside"]):
        tag.decompose()

    # Get the main content area if it exists, otherwise use body
    main = soup.find("main") or soup.find("body") or soup
    clean_html = str(main)

    return md(clean_html, heading_style="ATX", bullets="-").strip()


# ── Chunk text into ~1400-char pieces ─────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 1400) -> list[str]:
    """Split text into overlapping chunks of ~chunk_size characters."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) < chunk_size:
            current += p + "\n\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            current = p + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ── Stable chunk ID: hash of URL + content ───────────────────────────────────
def make_chunk_id(url: str, content: str) -> str:
    raw = f"{url}::{content[:200]}"
    return "chunk_" + hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Page content hash for fast change detection ───────────────────────────────
def page_hash(markdown: str) -> str:
    return hashlib.md5(markdown.encode()).hexdigest()


# ── Fetch a single page with Playwright ──
def fetch_page(page, url: str) -> tuple[str, str]:
    """
    Navigate to `url` using an already-open Playwright Page.
    Returns (title, markdown).  Raises on error.
    """
    page.goto(url, wait_until="domcontentloaded", timeout=15000)
    time.sleep(1.5) # wait for SPA render
    html  = page.content()
    title = page.title()
    markdown = html_to_markdown(html)
    return title, markdown


# ── Crawl the site and return {url: {title, markdown, hash}} ─────────────────
def crawl_site(limit: int = CRAWL_LIMIT) -> dict[str, dict]:
    """BFS crawl starting from TARGET_URL.  Uses a single Playwright browser session."""
    results: dict[str, dict] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.set_default_timeout(15000)
        
        try:
            # ── BFS crawl ─────────────────────────────────────────────────────────
            visited:  set[str] = set()
            to_visit: list[str] = [TARGET_URL]

            while to_visit and len(results) < limit:
                raw_url = to_visit.pop(0)
                url, _ = urldefrag(raw_url)   # strip #fragment
                url = url.rstrip('/')         # strip trailing slash to avoid duplicates
                if url in visited:
                    continue
                visited.add(url)

                try:
                    title, markdown = fetch_page(page, url)
                    results[url] = {
                        "title":    title,
                        "markdown": markdown,
                        "hash":     page_hash(markdown),
                    }
                    logger.info(f"[scraped] {url} ({len(markdown)} chars)")

                    # Collect internal links, stripping fragments to avoid duplicates
                    hrefs = page.eval_on_selector_all("a", "elements => elements.map(e => e.href)")
                    
                    for href in hrefs:
                        if not href:
                            continue
                        clean_href, _ = urldefrag(href)   # strip #fragment
                        clean_href = clean_href.rstrip('/')
                        if (
                            clean_href.startswith(TARGET_URL)
                            and clean_href not in visited
                            and clean_href not in to_visit
                            and len(results) + len(to_visit) < limit
                        ):
                            to_visit.append(clean_href)

                except PlaywrightTimeoutError:
                    logger.warning(f"[timeout] {url}")
                except Exception as e:
                    logger.error(f"[error] {url}: {e}")

        finally:
            browser.close()

    return results


# ── Load/save the per-page cache ──────────────────────────────────────────────
def load_previous_pages() -> dict[str, dict]:
    PAGES_FILE.parent.mkdir(parents=True, exist_ok=True)
    if PAGES_FILE.exists():
        try:
            with open(PAGES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_pages(pages: dict[str, dict]) -> None:
    PAGES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PAGES_FILE, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)


# ── Surgical ChromaDB update (only changed pages) ─────────────────────────────
def update_chromadb(changed_pages: dict[str, dict], removed_urls: set[str]) -> None:
    """Add/update chunks for changed pages and delete chunks for removed pages."""
    client   = chromadb.PersistentClient(path=str(DB_DIR))
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name="iryax_context_v2",
        embedding_function=embed_fn,
    )

    # ── Delete old chunks for changed / removed pages ─────────────────────────
    urls_to_delete = set(changed_pages.keys()) | removed_urls
    if urls_to_delete:
        try:
            existing = collection.get(where={"url": {"$in": list(urls_to_delete)}})
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
                logger.info(f"[ChromaDB] Deleted {len(existing['ids'])} old chunks for {len(urls_to_delete)} pages")
        except Exception as e:
            logger.warning(f"[ChromaDB] Warning during delete: {e}")

    # ── Add new chunks for changed pages ──────────────────────────────────────
    new_ids, docs, metas = [], [], []
    for url, page_data in changed_pages.items():
        for tc in chunk_text(page_data["markdown"]):
            new_ids.append(make_chunk_id(url, tc))
            docs.append(tc)
            metas.append({"url": url, "label": page_data["title"]})

    if new_ids:
        batch_size = 100
        for i in range(0, len(new_ids), batch_size):
            collection.add(
                ids=new_ids[i:i+batch_size],
                documents=docs[i:i+batch_size],
                metadatas=metas[i:i+batch_size],
            )
        logger.info(f"[ChromaDB] Added {len(new_ids)} new chunks for {len(changed_pages)} pages")


# ── Save full chunk snapshot ───────────────────────────────────────────────────
def save_chunks_snapshot(all_pages: dict[str, dict]) -> None:
    all_chunks = []
    for url, page_data in all_pages.items():
        for tc in chunk_text(page_data["markdown"]):
            all_chunks.append({
                "chunk_id": make_chunk_id(url, tc),
                "content":  tc,
                "url":      url,
                "label":    page_data["title"],
            })
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"[snapshot] {len(all_chunks)} chunks saved to {OUTPUT_FILE}")


# ── Main scrape-and-update job ────────────────────────────────────────────────
def scrape_job():
    logger.info("=" * 60)
    logger.info("Scrape job started")
    logger.info("=" * 60)

    # 1. Crawl site
    logger.info("[1] Crawling site with Playwright + BeautifulSoup...")
    current_pages = crawl_site()
    logger.info(f"Fetched {len(current_pages)} pages total.")

    if not current_pages:
        logger.error("[Error] 0 pages fetched (possible network timeout or site unreachable). Aborting update to protect database!")
        return

    # 2. Load previous page cache
    previous_pages = load_previous_pages()

    # 3. Detect changes at page level
    changed_pages: dict[str, dict] = {}
    removed_urls:  set[str]        = set()

    # New or updated pages
    for url, data in current_pages.items():
        prev = previous_pages.get(url)
        if prev is None or prev.get("hash") != data["hash"]:
            changed_pages[url] = data

    # Pages that no longer exist
    for url in previous_pages:
        if url not in current_pages:
            removed_urls.add(url)

    logger.info("[2] Change detection:")
    logger.info(f"    {len(changed_pages)} page(s) new/updated")
    logger.info(f"    {len(removed_urls)} page(s) removed")

    if not changed_pages and not removed_urls:
        logger.info("[3] No content changes detected. ChromaDB is up to date. [OK]")
    else:
        # 4. Re-chunk only changed pages and update ChromaDB surgically
        logger.info("[3] Updating ChromaDB (only changed pages)...")
        update_chromadb(changed_pages, removed_urls)

        # 5. Save updated page cache
        merged = {**previous_pages, **current_pages}
        for url in removed_urls:
            merged.pop(url, None)
        save_pages(merged)

        # 6. Save full chunk snapshot for reference
        logger.info("[4] Saving full chunk snapshot...")
        save_chunks_snapshot(merged)

        # 7. Invalidate semantic cache so AI returns fresh data immediately
        try:
            from rag_pipeline import clear_semantic_cache
            cleared = clear_semantic_cache()
            logger.info(f"[5] Invalidated semantic cache ({cleared} entries cleared).")
        except Exception as e:
            logger.warning(f"Could not clear semantic cache: {e}")

        # 8. Generate new fast-reply intents via LLM
        try:
            from generate_intents import generate_structured_intents
            logger.info(f"[6] Proactively updating fast-reply intents for {len(changed_pages)} changed page(s)...")
            generate_structured_intents(changed_pages)
        except Exception as e:
            logger.warning(f"Failed to generate structured intents: {e}")

    logger.info("[Done] Iryax AI Assistant has the latest knowledge.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_dotenv()

    # Run once immediately, then schedule every 10 minutes
    scrape_job()

    schedule.every(10).minutes.do(scrape_job)
    schedule.every(3).hours.do(clear_log_file)
    logger.info("Scheduler running. Scrape will repeat every 10 minutes. scraper.log will clear every 3 hours. Press Ctrl+C to stop.")

    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logger.error(f"[Scheduler Error] Unhandled exception in schedule execution: {e}")
        time.sleep(30)

