import sys
import os
import tempfile
import requests
from typing import List, Optional
# NEW: logging + colorama
import logging
from colorama import Fore, Style, init as colorama_init
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama


from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import faiss  # NEW

colorama_init(autoreset=True)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class RAGSystem:
    def __init__(
        self,
        llm_model: str = "qwen3:4b",
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        enable_faiss_gpu: bool = False,  # NEW
    ):
        self.llm = Ollama(model=llm_model, temperature=0.1, system="/set nothink")
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._vectorstore = None
        self._retriever = None
        self._memory: Optional[ConversationBufferMemory] = None
        self.enable_faiss_gpu = enable_faiss_gpu  # NEW
        self._qa_history: list[tuple[str, str]] = []  # NEW: lightweight history for prompt compression
        # NEW: simple banner
        self._log(f"Initialized RAGSystem (LLM={llm_model}, EMBED={embedding_model}, chunk={chunk_size}/{chunk_overlap})", Fore.CYAN)

    # NEW helper
    def _log(self, msg: str, color: str = Fore.WHITE):
        print(color + msg + Style.RESET_ALL)
        logger.info(msg)

    def _download_pdf(self, url_or_path: str) -> str:
        if os.path.isfile(url_or_path):
            self._log(f"Using local PDF: {url_or_path}", Fore.CYAN)
            return url_or_path
        self._log(f"Downloading PDF from URL: {url_or_path}", Fore.CYAN)
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        r = requests.get(url_or_path, timeout=60)
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(r.content)
        self._log(f"Download complete -> {tmp_path}", Fore.GREEN)
        return tmp_path

    def _build_retriever(self, pdf_path: str):
        self._log("Loading PDF pages...", Fore.CYAN)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        self._log(f"Loaded {len(docs)} document pages. Splitting into chunks...", Fore.CYAN)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separators=[
                "\n\n", "\n", ". ", "? ", "! ", " ", ""
            ]
        )
        splits = splitter.split_documents(docs)
        self._log(f"Created {len(splits)} text chunks. Building embeddings + FAISS index...", Fore.CYAN)
        self._vectorstore = FAISS.from_documents(splits, self.embeddings)
        # NEW: try GPU
        try:
            if self.enable_faiss_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self._vectorstore.index = faiss.index_cpu_to_gpu(res, 0, self._vectorstore.index)
                self._log("FAISS index moved to GPU.", Fore.GREEN)
            else:
                self._log("FAISS on CPU (set enable_faiss_gpu=True to try GPU).", Fore.CYAN)
        except Exception as e:
            self._log(f"FAISS GPU init failed, using CPU. Reason: {e}", Fore.RED)
        self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 4})
        self._log(f"Embedding + index build successful. Vectors stored: {ntotal}", Fore.GREEN)

    def create_vector_store(self, pdf_url: str, use_memory: bool = True) -> bool:
        """Download (if needed) and build retriever once; returns True on success."""
        pdf_path = self._download_pdf(pdf_url)
        try:
            self._build_retriever(pdf_path)
        finally:
            if not os.path.isfile(pdf_url) and os.path.exists(pdf_path):
                os.remove(pdf_path)
        if use_memory and self._memory is None:
            self._memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            self._log("Conversation memory initialized.", Fore.CYAN)
        self._log("Vector store ready. You can now ask questions.", Fore.GREEN)
        return True

    # NEW: constant system prompt to suppress CoT
    NOTHINK_SYSTEM_PROMPT = (
        "/set nothink\n"
        "You are a helpful assistant. Answer ONLY with the final answer. "
        "Do NOT show reasoning, chain-of-thought, analysis, steps, internal thoughts. "
        "Be very concise, to the point and factual."
        "Answer in a single sentence if possible."
    )

    # NEW helper: compact history string (last N)
    def _compact_history(self, limit: int = 4) -> str:
        if not self._qa_history:
            return ""
        recent = self._qa_history[-limit:]
        return "\n".join(f"Q: {q}\nA: {a}" for q, a in recent)

    # NEW helper: strip reasoning artifacts if model still leaks them
    def _strip_reasoning(self, text: str) -> str:
        markers = ["Thought:", "Thinking:", "Chain of Thought:", "Reasoning:", "Let's think", "Let's reason", "Let’s think"]
        lower = text.lower()
        for m in markers:
            idx = lower.find(m.lower())
            if idx != -1 and idx < len(text) * 0.6:  # early marker => likely reasoning prelude
                text = text[:idx].strip()
        # Remove trailing quotes / artifacts
        return text.strip().strip('"').strip()

    def answer_questions(self, questions: List[str]) -> List[str]:
        """Answer questions using existing retriever (must be created first) with strict no-CoT prompt."""
        if self._retriever is None:
            raise RuntimeError("Vector store not initialized. Call create_vector_store first.")
        answers = []
        for q in questions:
            self._log(f"Q: {q}", Fore.YELLOW)
            # Retrieve context
            docs = self._retriever.get_relevant_documents(q)
            context = "\n\n".join(d.page_content[:1000] for d in docs)  # truncate each doc to keep prompt small
            history_block = self._compact_history()
            prompt = (
                f"{self.NOTHINK_SYSTEM_PROMPT}\n\n"
                f"""{"Recent QA Pairs: " + history_block + " " if history_block else ''}"""
                f"Context:\n{context}\n\n"
                f"Question: {q}\n"
                f"Answer (final only):"
            )
            raw = self.llm.invoke(prompt)
            ans = self._strip_reasoning(raw)
            ans = ans.replace('<think>', '').replace('</think>', '').strip()
            answers.append(ans)
            self._qa_history.append((q, ans))
            trimmed = (ans[:180] + "...") if len(ans) > 200 else ans
            self._log(f"A: {trimmed}", Fore.MAGENTA)
        return answers

    def process_pdf_and_answer(
        self,
        pdf_url: str,
        questions: List[str],
        use_memory: bool = True
    ) -> List[str]:
        """Legacy one-shot helper: builds store then answers."""
        self._log("Starting end-to-end processing (PDF -> embeddings -> answers)", Fore.CYAN)
        self.create_vector_store(pdf_url, use_memory=use_memory)
        out = self.answer_questions(questions)
        self._log("Completed all questions.", Fore.GREEN)
        return out