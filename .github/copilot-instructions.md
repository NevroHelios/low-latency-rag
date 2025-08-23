# Low-Latency RAG System

Low-latency RAG (Retrieval-Augmented Generation) is a Python-based question-answering system that processes documents (PDFs, Word, PowerPoint, Excel) and provides fast responses using both local LLM models via Ollama and remote OpenAI models. The system is deployed as a FastAPI web service and includes a CLI interface for testing.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap, Build, and Test the Repository
- **Install uv package manager** (if not available): `pip install --user uv` -- takes 30-60 seconds
- Install Python dependencies:
  - **Primary method**: `uv sync` -- takes 1-2 minutes. NEVER CANCEL. Set timeout to 10+ minutes.
  - **Fallback method**: `pip install --user -r requirements.deploy.txt` -- takes 5-6 minutes. NEVER CANCEL. Set timeout to 15+ minutes.
- Validate installation: `python3 -c "import fastapi; import langchain; import numpy; print('All imports successful')"`
- Compile all Python files: `find . -path "./.venv" -prune -o -name "*.py" -exec python3 -m py_compile {} \;` -- takes <30 seconds

### Environment Setup (CRITICAL)
- **For OpenAI backend**: Export `OPENAI_API_KEY=your_actual_key` (required for OpenAI features)
- **For local testing**: Use `OPENAI_API_KEY=sk-fake-key` (app will start but OpenAI calls will fail)
- **For Ollama backend**: Ensure Ollama service is running locally with models: `qwen3:4b` and `nomic-embed-text`

### Run the Applications
- **FastAPI Web Server**:
  - Development: `uvicorn app:app --host 0.0.0.0 --port 8000 --reload` -- starts in 30-45 seconds. NEVER CANCEL.
  - Alternative: `python3 app.py` -- starts in 30-45 seconds. NEVER CANCEL.
  - Access at: http://localhost:8000 (API docs at /docs)
- **CLI Interface**:
  - Interactive mode: `python3 main.py --interactive`
  - Default demo: `python3 main.py` (uses hardcoded PDF and questions)

### Docker Deployment
- Build: `docker build -t low-latency-rag .` -- takes 5-10 minutes. NEVER CANCEL. Set timeout to 20+ minutes.
- Run: `docker run -p 8000:8000 -e OPENAI_API_KEY=your_key low-latency-rag`
- **Note**: Docker build may fail in restricted environments due to SSL certificate issues.

## Validation

### Manual Testing Requirements
After making any changes, ALWAYS validate with these scenarios:

#### FastAPI Server Validation (CRITICAL - Always Required)
1. Start server with mock key: `OPENAI_API_KEY=sk-fake-key python3 app.py` -- takes 60+ seconds. NEVER CANCEL.
2. **While server is running**, verify endpoints respond (in separate terminal):
   - `curl http://localhost:8000/` - should return `{"message": "RAG System API is running"}`
   - `curl http://localhost:8000/health` - should return `{"status": "healthy"}`
   - `curl http://localhost:8000/models` - should return model configuration JSON
3. Alternative startup method: `OPENAI_API_KEY=sk-fake-key uvicorn app:app --host 0.0.0.0 --port 8000 --reload`

#### CLI Interface Validation  
1. Test basic initialization: `python3 main.py` (will fail on PDF download, which is expected in restricted environments)
2. Test interactive mode start: `python3 main.py --interactive` (enter 'quit' at the PDF URL prompt to exit gracefully)

#### Code Quality Validation (Required Before Every Commit)
- Always validate Python syntax: `find . -path "./.venv" -prune -o -name "*.py" -exec python3 -m py_compile {} \;` -- takes <30 seconds
- Check imports work: `python3 -c "from app import app; from rag_system import RAGSystem; from rag_openai import RAGOpenAI; print('Imports OK')"`
- Basic dependency check: `python3 -c "import fastapi; import langchain; import numpy; print('All imports successful')"`

### Expected Behavior in Restricted Environments
- PDF downloads will fail with "No address associated with hostname" - this is normal
- Docker builds may fail with SSL certificate errors - document this limitation
- OpenAI API calls with fake keys will fail gracefully at runtime

## Common Tasks

### Key Project Structure
```
/home/runner/work/low-latency-rag/low-latency-rag/
├── app.py              # FastAPI web server (main entry point)
├── main.py             # CLI interface
├── rag_system.py       # Local Ollama-based RAG implementation
├── rag_openai.py       # OpenAI-based RAG implementation  
├── requirements.deploy.txt  # Pip dependencies (deployment)
├── pyproject.toml      # Project config (uv dependencies)
├── uv.lock            # Lockfile for uv package manager
├── Dockerfile         # Container build instructions
└── test.py            # Basic test imports
```

### Dual RAG Architecture
The system supports two backends:
1. **Local/Ollama Backend** (`rag_system.py`):
   - Uses local LLM models (default: `qwen3:4b`)
   - Embeddings via `nomic-embed-text`
   - Requires Ollama service running
   
2. **OpenAI Backend** (`rag_openai.py`):
   - Uses OpenAI GPT models (default: `gpt-5-mini`)  
   - Embeddings via `text-embedding-3-small`
   - Requires valid `OPENAI_API_KEY`

### Timing Expectations (CRITICAL - Set Proper Timeouts)
- **uv installation**: 30-60 seconds. NEVER CANCEL. Set timeout to 2+ minutes.
- **Package installation**: 5-6 minutes with pip, 1-2 minutes with uv. NEVER CANCEL. Set timeout to 15+ minutes.
- **App startup**: 60+ seconds including vector store initialization attempts. NEVER CANCEL. Set timeout to 2+ minutes.
- **Docker build**: 5-10 minutes in normal environments. NEVER CANCEL. Set timeout to 20+ minutes.
- **PDF processing**: 2-5 minutes per document depending on size (when network access available).
- **Basic validation tests**: <30 seconds each.

### Expected Startup Behavior
During FastAPI server startup, you will see:
1. Deprecation warning about `on_event` (expected, not an error)
2. Uvicorn starting on port 8000
3. RAG system initialization messages
4. Attempts to download known PDFs (will fail with "No address associated with hostname" in restricted environments)
5. "Application startup complete" message after ~60 seconds

### Common Commands Reference
```bash
# Quick validation
python3 -c "import fastapi; import langchain; print('Dependencies OK')"

# Start development server
OPENAI_API_KEY=sk-fake-key uvicorn app:app --reload --port 8000

# Test CLI
python3 main.py --interactive

# Syntax check all files  
find . -path "./.venv" -prune -o -name "*.py" -exec python3 -m py_compile {} \;

# Package management
pip install --user uv                     # Install uv if not available
uv sync                                    # Preferred (1-2 minutes)
pip install -r requirements.deploy.txt    # Fallback (5-6 minutes)
```

### Configuration Files
- **pyproject.toml**: Main project configuration, uv dependencies, Python >=3.10 requirement
- **requirements.deploy.txt**: Pip-compatible deployment dependencies (18 packages)
- **uv.lock**: Exact dependency versions for reproducible builds
- **.python-version**: Python 3.10 version specification

## Troubleshooting

### Common Issues
1. **"OPENAI_API_KEY not set"**: Export the environment variable or use `sk-fake-key` for testing
2. **Network/SSL errors**: Expected in restricted environments, document as limitation
3. **"No address associated with hostname"**: Normal when external PDF URLs are unreachable
4. **Import errors**: Run dependency installation commands with proper timeouts

### Development Tips
- Always test both RAG backends when making changes to core functionality
- Use mock API keys for development/testing when real keys unavailable
- The FastAPI app includes startup validation that pre-processes known PDF URLs
- Logs are written to `app.log` for debugging startup issues
- Vector stores use FAISS with optional GPU acceleration (falls back to CPU)

### Package Management Priority
1. Install uv if not available: `pip install --user uv` (30-60 seconds)
2. Try `uv sync` first (1-2 minutes, faster and more reliable)
3. Fall back to `pip install --user -r requirements.deploy.txt` if uv fails (5-6 minutes)
4. Both methods install the same functional dependencies
5. Always validate with import tests after installation