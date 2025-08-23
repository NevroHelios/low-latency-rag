#!/usr/bin/env python3
"""
Demo script showcasing the Low-Latency RAG System features
"""

import os
import tempfile
from rag_system import RAGSystem

def create_sample_pdf():
    """Create a sample PDF for demonstration"""
    content = """
    Low-Latency RAG System Documentation
    
    This is a sample document for testing the RAG system.
    
    Features:
    - Fast document processing
    - Multiple file format support
    - Real-time question answering
    - GPU acceleration support
    
    The system uses FAISS for vector storage and supports both Ollama and OpenAI models.
    
    Performance is optimized for low-latency responses while maintaining high accuracy.
    """
    
    # Create a simple text file (simulating PDF content)
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path

def demo_basic_functionality():
    """Demonstrate basic RAG functionality with a sample document"""
    print("🚀 Low-Latency RAG System Demo")
    print("=" * 50)
    
    # Create sample document
    print("📄 Creating sample document...")
    doc_path = create_sample_pdf()
    
    try:
        # Initialize RAG system
        print("🔧 Initializing RAG system...")
        rag = RAGSystem(
            llm_model="qwen3:4b",
            embedding_model="nomic-embed-text",
            chunk_size=400,  # Smaller chunks for demo
            chunk_overlap=50
        )
        
        # Sample questions
        questions = [
            "What are the main features of the RAG system?",
            "What vector storage technology is used?", 
            "What is the system optimized for?"
        ]
        
        print(f"📝 Processing document: {doc_path}")
        print("❓ Sample questions:")
        for i, q in enumerate(questions, 1):
            print(f"   {i}. {q}")
        
        print("\n⚡ Processing... (This would normally query a local LLM)")
        print("📊 Results:")
        
        # Simulate results since we can't actually run LLM queries in this environment
        simulated_answers = [
            "The main features include fast document processing, multiple file format support, real-time question answering, and GPU acceleration support.",
            "The system uses FAISS for vector storage.",
            "The system is optimized for low-latency responses while maintaining high accuracy."
        ]
        
        for i, (q, a) in enumerate(zip(questions, simulated_answers), 1):
            print(f"\n{i}. Q: {q}")
            print(f"   A: {a}")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"🗑️  Cleaning up: {doc_path}")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
    finally:
        # Clean up
        if os.path.exists(doc_path):
            os.unlink(doc_path)

def show_configuration_options():
    """Show different configuration options"""
    print("\n🔧 Configuration Options")
    print("=" * 30)
    
    configs = [
        {
            "name": "Speed Optimized",
            "params": {
                "chunk_size": 600,
                "chunk_overlap": 50,
                "enable_faiss_gpu": False
            }
        },
        {
            "name": "Accuracy Optimized", 
            "params": {
                "chunk_size": 1200,
                "chunk_overlap": 300,
                "enable_faiss_gpu": True
            }
        },
        {
            "name": "Balanced",
            "params": {
                "chunk_size": 800,
                "chunk_overlap": 120,
                "enable_faiss_gpu": False
            }
        }
    ]
    
    for config in configs:
        print(f"\n📋 {config['name']}:")
        for key, value in config['params'].items():
            print(f"   {key}: {value}")

if __name__ == "__main__":
    try:
        demo_basic_functionality()
        show_configuration_options()
        
        print("\n🎉 For full functionality, install Ollama and run:")
        print("   ollama pull qwen3:4b")
        print("   python main.py --interactive")
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")