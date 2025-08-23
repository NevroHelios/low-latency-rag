# Low-Latency RAG System

A Retrieval-Augmented Generation (RAG) system that provides both a FastAPI web service and CLI interface for document processing and question answering. The system supports both local Ollama models and OpenAI models.

**CRITICAL**: Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Dependencies
**NEVER CANCEL**: Dependency installation takes 30-60 seconds. Set timeout to 120+ seconds.
```bash
# Install Python dependencies (takes 30-60 seconds)
pip install -r requirements.deploy.txt

# Verify core imports work (takes <1 second)
python3 -c "from rag_system import RAGSystem; from rag_openai import RAGOpenAI; print('Imports successful')"
```

### External Service Requirements
This application **REQUIRES** external services to function fully:

#### Ollama Setup (for rag_system.py - LangChain implementation)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models - NEVER CANCEL: Each model download takes 5-15 minutes
ollama pull qwen3:4b        # LLM model (takes 5-15 minutes)
ollama pull nomic-embed-text # Embedding model (takes 2-5 minutes)

# Start Ollama service (runs on localhost:11434)
ollama serve
```

#### OpenAI Setup (for rag_openai.py - OpenAI implementation)
```bash
# Set OpenAI API key environment variable
export OPENAI_API_KEY="your-openai-api-key-here"
```

### Running the Application

#### CLI Interface
```bash
# Default mode with hardcoded test PDF - NEVER CANCEL: Takes 60+ seconds for full processing
python3 main.py

# Interactive mode for custom PDFs
python3 main.py --interactive
```

#### Web Server
```bash
# Start FastAPI server - NEVER CANCEL: Startup takes 30-120 seconds depending on vector store initialization
python3 app.py
# Server runs on http://0.0.0.0:8000
```

#### Docker Deployment
```bash
# Build Docker image - NEVER CANCEL: Takes 5-10 minutes
# NOTE: May fail in environments with SSL/certificate restrictions
docker build -t rag-system .

# Run container (after successful build)
docker run -p 8000:8000 rag-system
```

## Validation

### Basic Validation (No External Services Required)
```bash
# Test dependency installation and imports (takes <5 seconds)
pip install -r requirements.deploy.txt
python3 -c "from rag_system import RAGSystem; from rag_openai import RAGOpenAI; print('Core imports work')"

# Verify FastAPI can import (takes <2 seconds)
python3 -c "from app import app; print('FastAPI app imports successfully')"
```

### Full Functionality Validation (Requires External Services)
**Prerequisites**: Ollama running with models downloaded, or OPENAI_API_KEY set

```bash
# Test CLI with local PDF - NEVER CANCEL: Takes 60-300 seconds for full PDF processing
# Note: Create a test PDF first for validation
python3 -c "
from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(0, 10, 'Test Document for RAG System', ln=True)
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, 'This is a test document for validation.', ln=True)
pdf.output('test_validation.pdf')
print('Created test_validation.pdf')
"

# Test server endpoints (server must be running)
curl -X GET http://localhost:8000/          # Should return: {"message": "RAG System API is running"}
curl -X GET http://localhost:8000/health    # Should return: {"status": "healthy"}
curl -X GET http://localhost:8000/models    # Should return model information
```

### Manual Testing Scenarios
**Always run these scenarios after making changes to validate functionality:**

1. **CLI Test**: Run `python3 main.py` and verify it processes PDF and returns answers
2. **Server Test**: Start server and test all endpoints return expected responses
3. **PDF Processing**: Verify the system can load, chunk, and embed documents
4. **Question Answering**: Test with sample questions to ensure RAG pipeline works

## Common Tasks

### Development Commands
```bash
# Run specific modules for testing
python3 -c "from rag_system import RAGSystem; rag = RAGSystem(); print('LangChain RAG initialized')"
python3 -c "from rag_openai import RAGOpenAI; print('OpenAI RAG module imported')"

# Check application logs
tail -f app.log
```

### Troubleshooting
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Verify OpenAI API key is set
echo $OPENAI_API_KEY

# Test PDF processing without external URLs (create local PDF first)
python3 main.py --interactive  # Then provide local file path
```

## File Structure Reference
```
├── app.py                 # FastAPI web service (main server)
├── main.py               # CLI interface 
├── rag_system.py         # LangChain + Ollama RAG implementation
├── rag_openai.py         # OpenAI-based RAG implementation
├── requirements.deploy.txt # Production dependencies
├── pyproject.toml        # Project configuration
├── Dockerfile            # Container configuration
└── test.py               # Test file (requires docling - not in requirements)
```

## Important Notes

### Timing Expectations
- **Dependency installation**: 30-60 seconds - NEVER CANCEL, set timeout to 120+ seconds
- **Model downloads**: 5-15 minutes per model - NEVER CANCEL, set timeout to 30+ minutes
- **Server startup**: 30-120 seconds - NEVER CANCEL, set timeout to 180+ seconds  
- **PDF processing**: 60-300 seconds depending on document size - NEVER CANCEL, set timeout to 600+ seconds
- **Basic imports**: <5 seconds

### Service Dependencies
- **CRITICAL**: Application cannot run without either Ollama (for local models) or OpenAI API key
- Default models: `qwen3:4b` (LLM) and `nomic-embed-text` (embeddings) for Ollama
- OpenAI models: `gpt-5-mini` (LLM) and `text-embedding-3-small` (embeddings)
- Server requires successful vector store initialization during startup

### Known Limitations
- `test.py` imports `docling` which is not in requirements.deploy.txt - will fail
- No linting tools configured (no flake8, black, or pytest in dependencies)
- No CI/CD pipeline configured
- Docker build may fail in environments with SSL/certificate restrictions
- Application fails gracefully when external services are unavailable
- Some test PDFs may be behind network restrictions in development environments

### Error Handling
- PDF download failures: Check network connectivity and URL validity
- Ollama connection errors: Ensure `ollama serve` is running on localhost:11434
- OpenAI errors: Verify OPENAI_API_KEY environment variable is set correctly
- Import errors: Run `pip install -r requirements.deploy.txt` to ensure dependencies

### Development Best Practices
- Always test both CLI and web interfaces after making changes
- Verify PDF processing works with various document types
- Test question-answering quality with sample documents
- Monitor app.log for detailed operation logs
- Use local test PDFs when external URLs are not accessible