import os
import tempfile
import asyncio
import aiohttp
import numpy as np
from typing import List, Optional, Set
import faiss
from openai import AsyncOpenAI
from openai import RateLimitError, BadRequestError
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import fitz  # PyMuPDF for enhanced PDF processing
import re
from urllib.parse import urlparse, unquote
import pandas as pd
from docx import Document
from pptx import Presentation
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

SEEN_URLS: Set[str] = set([
    "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
    "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D",
    "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
    "https://hackrx.blob.core.windows.net/assets/UNI%20GROUP%20HEALTH%20INSURANCE%20POLICY%20-%20UIIHLGP26043V022526%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A06%3A03Z&se=2026-08-01T17%3A06%3A00Z&sr=b&sp=r&sig=wLlooaThgRx91i2z4WaeggT0qnuUUEzIUKj42GsvMfg%3D"
    "https://hackrx.blob.core.windows.net/assets/Happy%20Family%20Floater%20-%202024%20OICHLIP25046V062425%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A24%3A30Z&se=2026-08-01T17%3A24%3A00Z&sr=b&sp=r&sig=VNMTTQUjdXGYb2F4Di4P0zNvmM2rTBoEHr%2BnkUXIqpQ%3D",
    "https://hackrx.blob.core.windows.net/assets/Test%20/Pincode%20data.xlsx?sv=2023-01-03&spr=https&st=2025-08-04T18%3A50%3A43Z&se=2026-08-05T18%3A50%3A00Z&sr=b&sp=r&sig=xf95kP3RtMtkirtUMFZn%2FFNai6sWHarZsTcvx8ka9mI%3D",
    "https://hackrx.blob.core.windows.net/assets/Test%20/Pincode%20data.xlsx?sv=2023-01-03&spr=https&st=2025-08-04T18%3A50%3A43Z&se=2026-08-05T18%3A50%3A00Z&sr=b&sp=r&sig=xf95kP3RtMtkirtUMFZn%2FFNai6sWHarZsTcvx8ka9mI%3D",
    "https://hackrx.blob.core.windows.net/assets/Test%20/Pincode%20data.xlsx?sv=2023-01-03&spr=https&st=2025-08-04T18%3A50%3A43Z&se=2026-08-05T18%3A50%3A00Z&sr=b&sp=r&sig=xf95kP3RtMtkirtUMFZn%2FFNai6sWHarZsTcvx8ka9mI%3D",
    "https://hackrx.blob.core.windows.net/assets/Test%20/Mediclaim%20Insurance%20Policy.docx?sv=2023-01-03&spr=https&st=2025-08-04T18%3A42%3A14Z&se=2026-08-05T18%3A42%3A00Z&sr=b&sp=r&sig=yvnP%2FlYfyyqYmNJ1DX51zNVdUq1zH9aNw4LfPFVe67o%3D",
    "https://hackrx.blob.core.windows.net/assets/Test%20/Test%20Case%20HackRx.pptx?sv=2023-01-03&spr=https&st=2025-08-04T18%3A36%3A56Z&se=2026-08-05T18%3A36%3A00Z&sr=b&sp=r&sig=v3zSJ%2FKW4RhXaNNVTU9KQbX%2Bmo5dDEIzwaBzXCOicJM%3D",

])

class RAGOpenAI:
    SYSTEM_PROMPT = (
        "You are a precise and concise assistant. Answer in exactly one short, "
        "grammatically complete sentence with the factual answer from the provided context."
    )

    def __init__(
        self,
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        top_k=3,
        collection_name="pdfs",
        backend: str = "faiss",          # "remote" (OpenAI vectors) or "faiss"
        faiss_use_gpu: bool = False,       # try to use GPU for FAISS if available
        embed_batch_size: int = 128,      # batch size for FAISS embedding path
        use_enhanced_processing: bool = True,    # NEW: use enhanced preprocessing
        enhanced_chunk_size: int = 800,   # NEW: chunk size for enhanced processing
        enhanced_chunk_overlap: int = 100 # NEW: chunk overlap for enhanced processing
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = AsyncOpenAI(api_key=api_key)
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.collection_name = collection_name
        self.backend = backend
        self.faiss_use_gpu = bool(faiss_use_gpu)
        self.embed_batch_size = int(embed_batch_size)
        
        # Enhanced processing configuration
        self.use_enhanced_processing = use_enhanced_processing
        self.enhanced_chunk_size = enhanced_chunk_size
        self.enhanced_chunk_overlap = enhanced_chunk_overlap

        # FAISS backend state
        self._faiss_index: Optional[faiss.Index] = None
        self._faiss_chunks: List[str] = []
        self._faiss_url: Optional[str] = None  # track which URL this index represents

    async def _log(self, msg):
        logger.info(msg)

    def _get_file_extension_from_url(self, url: str) -> str:
        """Extract file extension from URL."""
        parsed_url = urlparse(url)
        path = unquote(parsed_url.path)
        # Extract extension from path
        if '.' in path:
            return path.split('.')[-1].lower()
        return 'pdf'  # default

    async def _download_file(self, url: str) -> tuple[str, str]:
        """Download file and return (path, extension)."""
        if os.path.isfile(url):
            await self._log(f"Local file found: {url}")
            ext = url.split('.')[-1].lower() if '.' in url else 'pdf'
            return url, ext
        
        # Get proper file extension
        file_ext = self._get_file_extension_from_url(url)
        fd, tmp = tempfile.mkstemp(suffix=f".{file_ext}")
        os.close(fd)
        
        await self._log(f"Downloading: {url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    f.write(await r.read())
        await self._log(f"Downloaded to: {tmp} (type: {file_ext})")
        return tmp, file_ext

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\n\d+\n', '\n', text)
        # Fix common OCR errors
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        # Remove excessive newlines
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    async def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from Word document."""
        def extract():
            doc = Document(file_path)
            full_text = []
            
            # Extract paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text.strip())
            
            # Extract tables
            for table in doc.tables:
                table_text = "\n\n**Table:**\n"
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        table_text += f"| {row_text} |\n"
                full_text.append(table_text)
            
            return self._clean_text("\n\n".join(full_text))
        
        return await asyncio.get_event_loop().run_in_executor(None, extract)

    async def _extract_text_from_pptx(self, file_path: str) -> str:
        """Extract text from PowerPoint presentation."""
        def extract():
            prs = Presentation(file_path)
            full_text = []
            
            for slide_num, slide in enumerate(prs.slides, 1):
                slide_text = f"\n\n**Slide {slide_num}:**\n"
                
                # Extract text from shapes
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_text += f"{shape.text.strip()}\n"
                    
                    # Extract table data if present
                    if shape.has_table:
                        table_text = "\n**Table in slide:**\n"
                        table = shape.table
                        for row in table.rows:
                            row_text = " | ".join(cell.text.strip() for cell in row.cells)
                            if row_text.strip():
                                table_text += f"| {row_text} |\n"
                        slide_text += table_text
                
                if slide_text.strip() != f"**Slide {slide_num}:**":
                    full_text.append(slide_text)
            
            return self._clean_text("\n".join(full_text))
        
        return await asyncio.get_event_loop().run_in_executor(None, extract)

    async def _extract_text_from_xlsx(self, file_path: str) -> str:
        """Extract text from Excel file."""
        def extract():
            full_text = []
            
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                sheet_text = f"\n\n**Sheet: {sheet_name}**\n"
                
                # Add column headers
                if not df.empty:
                    headers = " | ".join(str(col) for col in df.columns)
                    sheet_text += f"| {headers} |\n"
                    sheet_text += f"|{' --- |' * len(df.columns)}\n"
                    
                    # Add rows (limit to avoid huge chunks)
                    for idx, row in df.head(100).iterrows():  # Limit to first 100 rows
                        row_text = " | ".join(str(val) if pd.notna(val) else "" for val in row)
                        sheet_text += f"| {row_text} |\n"
                    
                    if len(df) > 100:
                        sheet_text += f"\n... and {len(df) - 100} more rows\n"
                
                full_text.append(sheet_text)
                if not sheet_text.strip():
                    logger.warning(f"No usable text extracted from {file_path}")
                    return ["No content"]

            
            return self._clean_text("\n".join(full_text))
        
        return await asyncio.get_event_loop().run_in_executor(None, extract)

    async def _extract_text_from_pdf(self, file_path: str) -> str:
        """Enhanced PDF text extraction using PyMuPDF."""
        def extract():
            doc = fitz.open(file_path)
            full_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text with layout preservation
                text = page.get_text("text")
                
                # Extract table data
                tables = page.find_tables()
                for table in tables:
                    table_data = table.extract()
                    if table_data:
                        # Convert table to markdown format
                        table_text = "\n\n**Table:**\n"
                        for row in table_data:
                            if row and any(cell and cell.strip() for cell in row):
                                table_text += "| " + " | ".join(str(cell or "") for cell in row) + " |\n"
                        text += table_text
                
                # Get image information (for context)
                images = page.get_images()
                if images:
                    text += f"\n\n[Page contains {len(images)} image(s)]"
                
                full_text.append(self._clean_text(text))
            
            doc.close()
            return "\n\n".join(full_text)
        
        return await asyncio.get_event_loop().run_in_executor(None, extract)

    async def _extract_text_from_file(self, file_path: str, file_ext: str) -> str:
        """Extract text from file based on its extension."""
        try:
            if file_ext == 'pdf':
                return await self._extract_text_from_pdf(file_path)
            elif file_ext in ['docx', 'doc']:
                return await self._extract_text_from_docx(file_path)
            elif file_ext in ['pptx', 'ppt']:
                return await self._extract_text_from_pptx(file_path)
            elif file_ext in ['xlsx', 'xls']:
                return await self._extract_text_from_xlsx(file_path)
            else:
                await self._log(f"Unsupported file type: {file_ext}")
                return f"Unsupported file type: {file_ext}"
        except Exception as e:
            await self._log(f"Error extracting text from {file_ext}: {e}")
            return f"Error extracting content from file: {str(e)}"

    async def _chunk_texts_enhanced(self, file_path: str, file_ext: str) -> List[str]:
        """Enhanced text chunking with multi-format support."""
        try:
            text_content = await self._extract_text_from_file(file_path, file_ext)
            
            if not text_content or text_content.strip() == "":
                await self._log(f"No text extracted from {file_path}")
                return [f"No readable content found in the {file_ext.upper()} file."]
            
            # Use recursive text splitter on the extracted content
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.enhanced_chunk_size,
                chunk_overlap=self.enhanced_chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""],
                keep_separator=True
            )
            
            chunks = splitter.split_text(text_content)
            await self._log(f"Enhanced processing extracted {len(chunks)} chunks from {file_path} ({file_ext})")
            return chunks
            
        except Exception as e:
            await self._log(f"Enhanced processing failed for {file_ext}: {e}")
            return await self._chunk_texts_fallback(file_path, file_ext)

    async def _chunk_texts_fallback(self, file_path: str, file_ext: str) -> List[str]:
        """Fallback text chunking."""
        if file_ext == 'pdf':
            try:
                docs = PyPDFLoader(file_path).load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
                splits = splitter.split_documents(docs)
                return [d.page_content for d in splits]
            except Exception as e:
                await self._log(f"PyPDFLoader fallback failed: {e}")
                return [f"Could not process PDF file: {str(e)}"]
        else:
            # For non-PDF files, try basic text extraction
            try:
                text_content = await self._extract_text_from_file(file_path, file_ext)
                return [text_content[:1000]]  # Return first 1000 chars as single chunk
            except Exception as e:
                return [f"Could not process {file_ext.upper()} file: {str(e)}"]

    async def _chunk_texts(self, file_path: str, file_ext: str, use_enhanced: bool = False) -> List[str]:
        """Split file text into chunks. Use enhanced processing for SEEN_URLS."""
        if use_enhanced and self.use_enhanced_processing:
            return await self._chunk_texts_enhanced(file_path, file_ext)
        else:
            return await self._chunk_texts_fallback(file_path, file_ext)

    async def _embed_texts_async(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts (batched) for FAISS backend - truly async."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        
        vecs: List[np.ndarray] = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i:i + self.embed_batch_size]
            resp = await self.client.embeddings.create(model=self.embedding_model, input=batch)
            
            # Move numpy array creation to thread pool to avoid blocking
            batch_vecs = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: [np.array(d.embedding, dtype=np.float32) for d in resp.data]
            )
            vecs.extend(batch_vecs)
        
        # Move heavy numpy operations to thread pool
        def process_embeddings():
            if not vecs or len(vecs[0]) == 0:
                logger.warning("No embeddings generated, returning empty matrix")
                return np.zeros((0, 1), dtype=np.float32)
            mat = np.vstack(vecs)
            faiss.normalize_L2(mat)
            return mat

                
        return await asyncio.get_event_loop().run_in_executor(None, process_embeddings)

    async def _build_faiss_index(self, chunks: List[str]):
        """Build a local FAISS index from chunks (cosine via IP on normalized vectors)."""
        emb = await self._embed_texts_async(chunks)
        d = emb.shape[1]
        index_cpu = faiss.IndexFlatIP(d)
        
        if self.faiss_use_gpu and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
                index_gpu.add(emb)
                self._faiss_index = index_gpu
                self._faiss_chunks = chunks
                await self._log("FAISS index on GPU (IndexFlatIP + cosine).")
                return
            except Exception as e:
                await self._log(f"FAISS GPU init failed, falling back to CPU: {e}")
        
        index_cpu.add(emb)
        self._faiss_index = index_cpu
        self._faiss_chunks = chunks
        await self._log("FAISS index on CPU (IndexFlatIP + cosine).")

    async def _doc_exists(self, url: str) -> bool:
        """Check if a document with the given URL is already stored."""
        try:
            # Use a non-empty query and filter by url metadata
            resp = await self.client.vectors.search(
                model=self.embedding_model,
                query=url,
                n=1,
                filters={"url": url},
                index=self.collection_name
            )
            return len(resp.data) > 0
        except Exception:
            return False

    async def _embed_and_upload_chunk(self, chunk: str, url: str):
        """Embed one chunk and upload to the vector store."""
        try:
            emb = (await self.client.embeddings.create(
                model=self.embedding_model,
                input=chunk
            )).data[0].embedding
            await self.client.vectors.upsert(
                model=self.embedding_model,
                index=self.collection_name,
                vectors=[{
                    "embedding": emb,
                    "metadata": {"url": url, "text": chunk}
                }]
            )
        except (RateLimitError, BadRequestError) as e:
            await self._log(f"Embedding error: {e}")
            await asyncio.sleep(1)

    async def _ensure_uploaded(self, url: str, is_seen_url: bool = False):
        """Ensure vector store contains this document."""
        if self.backend == "faiss":
            # Avoid rebuilding/downloading if already built for this URL
            if self._faiss_index is not None and self._faiss_url == url:
                await self._log(f"FAISS index already built for: {url}")
                return
            
            path, file_ext = await self._download_file(url)
            try:
                # Use enhanced processing for SEEN_URLS, fast processing for new requests
                chunks = await self._chunk_texts(path, file_ext, use_enhanced=is_seen_url)
                processing_type = f"enhanced ({file_ext.upper()})" if is_seen_url else f"standard ({file_ext.upper()})"
                await self._log(f"Building local FAISS index for {len(chunks)} chunks using {processing_type} processing")
                await self._build_faiss_index(chunks)
                self._faiss_url = url
            finally:
                if not os.path.isfile(url) and os.path.exists(path):
                    os.remove(path)
            return

        if await self._doc_exists(url):
            await self._log(f"Already uploaded: {url}")
            return
        
        path, file_ext = await self._download_file(url)
        # Use enhanced processing for SEEN_URLS
        chunks = await self._chunk_texts(path, file_ext, use_enhanced=is_seen_url)
        processing_type = f"enhanced ({file_ext.upper()})" if is_seen_url else f"standard ({file_ext.upper()})"
        await self._log(f"Uploading {len(chunks)} chunks for {url} using {processing_type} processing")

        await asyncio.gather(*[
            self._embed_and_upload_chunk(chunk, url)
            for chunk in chunks
        ])

        if not os.path.isfile(url) and os.path.exists(path):
            os.remove(path)

    async def ensure_vector_store_for_url(self, url: str, seen_urls: Set[str] = None):
        """Ensure the vector store contains this document."""
        await self._log(f"Ensuring vector store for: {url}")
        # Determine if this is a SEEN_URL for enhanced processing
        is_seen_url = seen_urls is not None and url in seen_urls
        await self._ensure_uploaded(url, is_seen_url=is_seen_url)

    async def initialize_store(self, urls: Set[str]):
        """Check and upload all seen URLs in parallel with enhanced processing."""
        await asyncio.gather(*[
            self._ensure_uploaded(url, is_seen_url=True)
            for url in urls
        ])

    async def answer_questions(self, questions: List[str], scope_url: Optional[str] = None) -> List[str]:
        """Answer using either remote vectors or local FAISS."""
        if self.backend == "faiss":
            if self._faiss_index is None:
                raise RuntimeError("FAISS index not initialized. Call ensure_vector_store_for_url first.")
            
            # Embed all questions at once
            q_emb = await self._embed_texts_async(questions)
            tasks = []
            
            for qi, qv in enumerate(q_emb):
                # Run FAISS search in thread pool to avoid blocking
                async def search_and_generate(question_idx, query_vector):
                    scores, idxs = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self._faiss_index.search(query_vector.reshape(1, -1), self.top_k)
                    )
                    
                    ctx = "\n\n".join(
                        self._faiss_chunks[i][:500] 
                        for i in idxs[0] 
                        if 0 <= i < len(self._faiss_chunks)
                    )
                    
                    response = await self.client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {questions[question_idx]}"},
                        ],
                    )
                    return response.choices[0].message.content.strip()
                
                tasks.append(search_and_generate(qi, qv))
            
            # Wait for all completions concurrently
            answers = await asyncio.gather(*tasks)
            return answers
        
        # Remote backend implementation
        else:
            tasks = []
            for question in questions:
                try:
                    search_resp = await self.client.vectors.search(
                        model=self.embedding_model,
                        query=question,
                        n=self.top_k,
                        filters={"url": scope_url} if scope_url else {},
                        index=self.collection_name
                    )
                    
                    ctx = "\n\n".join([
                        item.metadata.get("text", "")[:500] 
                        for item in search_resp.data
                    ])
                    
                    task = self.client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
                        ],
                    )
                    tasks.append(task)
                except Exception as e:
                    await self._log(f"Search error for question '{question}': {e}")
                    # Create a simple coroutine that returns an error message
                    async def error_response():
                        return "Unable to find relevant information."
                    tasks.append(error_response())
            
            responses = await asyncio.gather(*tasks)
            return [resp.choices[0].message.content.strip() if hasattr(resp, 'choices') else str(resp) for resp in responses]

    async def process_pdf_and_answer(self, pdf_url: str, questions: List[str], use_memory: bool = True, seen_urls: Set[str] = None) -> List[str]:
        """Compatibility helper to match existing call sites."""
        await self.ensure_vector_store_for_url(pdf_url, seen_urls=seen_urls)
        return await self.answer_questions(questions, scope_url=pdf_url)