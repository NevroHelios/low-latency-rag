from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import asyncio
from rag_system import RAGSystem
from rag_openai import RAGOpenAI, SEEN_URLS  # import SEEN_URLS

# Ensure logs go to app.log, opened in a+ mode, and include INFO
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
log_file_pointer = logging.FileHandler("app.log", mode="a+", encoding="utf-8")
log_file_pointer.setLevel(logging.INFO)
logger.addHandler(log_file_pointer)

LLM_MODEL = "qwen3:4b"
EMBED_MODEL = "nomic-embed-text"
# NEW: OpenAI defaults
OPENAI_LLM_MODEL = "gpt-5-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

# Fill known URLs so we can pre-create vector stores at startup
KNOWN_URLS_LIST = SEEN_URLS
KNOWN_URLS_SET = set(KNOWN_URLS_LIST)

app = FastAPI(title="RAG System API", description="API for document processing and question answering using RAG", version="1.0.0")

security = HTTPBearer()
EXPECTED_TOKEN = "780ce402dcefc8c9d0601379a4e85660dff21fbfdf19d6d3d6aa8e0d081f78ad"

class RAGRequest(BaseModel):
    documents: str
    questions: List[str]

class RAGResponse(BaseModel):
    answers: List[str]

# One global scratch RAG system (for unknown URLs, can overwrite its index).
_default_rag: Optional[RAGSystem] = None
# Dedicated RAG systems for the two predefined URLs (separate indexes).
_rag_by_url: Dict[str, RAGSystem] = {}
# NEW: parallel caches for OpenAI
_default_rag_openai: Optional[RAGOpenAI] = None
_rag_openai_by_url: Dict[str, RAGOpenAI] = {}

# Global semaphore to serialize heavy work and protect limited resources.
# Handles "spamming" by processing requests one-by-one.
_compute_lock = asyncio.Semaphore(1)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token", headers={"WWW-Authenticate": "Bearer"})
    return credentials.credentials

def _new_rag() -> RAGSystem:
    logger.info("Init RAG system (LangChain + Ollama)")
    return RAGSystem(llm_model=LLM_MODEL, embedding_model=EMBED_MODEL)

# NEW
def _new_openai_rag() -> RAGOpenAI:
    logger.info("Init RAG system (OpenAI)")
    return RAGOpenAI(llm_model=OPENAI_LLM_MODEL, embedding_model=OPENAI_EMBED_MODEL, backend="faiss")

def get_rag_for_url(url: Optional[str]) -> RAGSystem:
    global _default_rag
    if url and url in _rag_by_url:
        return _rag_by_url[url]
    if _default_rag is None:
        _default_rag = _new_rag()
    return _default_rag

# NEW
def get_openai_rag_for_url(url: Optional[str]) -> RAGOpenAI:
    global _default_rag_openai
    if url and url in _rag_openai_by_url:
        return _rag_openai_by_url[url]
    if _default_rag_openai is None:
        _default_rag_openai = _new_openai_rag()
    return _default_rag_openai


@app.on_event("startup")
async def _startup():
    # Create dedicated OpenAI-backed RAG systems for the predefined URLs
    for url in KNOWN_URLS_LIST:
        _rag_openai_by_url[url] = _new_openai_rag()
    # Ensure OpenAI Vector Stores exist (create+upload if missing)
    for url in KNOWN_URLS_LIST:
        try:
            async with _compute_lock:
                logger.info("Ensuring OpenAI Vector Store for: %s", url)
                # Use enhanced processing for known URLs
                await _rag_openai_by_url[url].ensure_vector_store_for_url(url, seen_urls=SEEN_URLS)
                logger.info("Vector Store ready for: %s", url)
        except Exception as e:
            logger.warning("Pre-provision failed for %s: %s", url, e)



@app.get("/")
async def root():
    return {"message": "RAG System API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.websocket("/api/v1/hackrx/run")
async def hackrx_run_ws_openai(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if token != EXPECTED_TOKEN:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    logger.info("WebSocket client connected to /hackrx/run/openai")
    try:
        while True:
            data = await websocket.receive_json()
            documents = data.get("documents")
            questions = data.get("questions")
            if not documents or not questions:
                await websocket.send_json({"error": "documents (str) and questions (list[str]) required"})
                continue
            try:
                rag = get_openai_rag_for_url(documents if documents in KNOWN_URLS_SET else None)
                async with _compute_lock:
                    # Ensure vector store exists then answer, scoped to this document
                    await rag.ensure_vector_store_for_url(documents)
                    answers = await rag.answer_questions(questions, scope_url=documents)
                if len(answers) != len(questions):
                    logger.warning("Answer count mismatch")
                # Log the questions and full answers to app.log
                logger.info("WS Q/A | documents=%s | questions=%s | answers=%s", documents, questions, answers)
                await websocket.send_json({"answers": answers})
            except Exception as e:
                logger.error(f"Unexpected error (openai): {e}")
                await websocket.send_json({"error": "internal_server_error"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from /hackrx/run/openai")

@app.post("/api/v1/hackrx/run", response_model=RAGResponse)
async def hackrx_webhook_openai(
    request: RAGRequest,
    token: str = Depends(verify_token)
):
    """
    HTTP POST webhook variant using async OpenAI Vector Store RAG.
    Body: { "documents": "<pdf>", "questions": ["..."] }
    """
    if not request.documents or not request.questions:
        raise HTTPException(
            status_code=400,
            detail="documents (str) and questions (list[str]) required"
        )
    try:
        # reuse cached instance (pre-provisioned for known URLs)
        rag = get_openai_rag_for_url(request.documents if request.documents in KNOWN_URLS_SET else None)
        async with _compute_lock:
            # Pass SEEN_URLS to enable enhanced processing for known URLs
            await rag.ensure_vector_store_for_url(request.documents, seen_urls=SEEN_URLS)
        answers = await rag.answer_questions(request.questions, scope_url=request.documents)
        if len(answers) != len(request.questions):
            logger.warning("Answer count mismatch (openai %d vs %d)", len(answers), len(request.questions))
        # Log the questions and full answers to app.log
        logger.info("HTTP Q/A | documents=%s | questions=%s | answers=%s", request.documents, request.questions, answers)
        return RAGResponse(answers=answers)
    except Exception as e:
        logger.exception("Webhook (openai) failed: %s", e)
        raise HTTPException(status_code=500, detail="internal_server_error")



@app.get("/models")
async def get_model_info():
    return {
        "llm_model": LLM_MODEL,
        "llm_provider": "Openai",
        "embedding_model": EMBED_MODEL,
        "embedding_provider": "Openai",
        "system_prompt": "/set nothink"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True,)