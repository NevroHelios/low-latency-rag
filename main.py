from rag_system import RAGSystem
from colorama import Fore

MODEL_NAME = "qwen3:4b"

def main():
    """
    Example usage of the RAG system.
    """
    print("Initializing RAG system...")

    # Initialize RAG system with qwen3:4b model
    rag = RAGSystem(llm_model=MODEL_NAME)

    # Example PDF URL (replace with actual PDF URL)
    pdf_url = None # input("Enter PDF URL: ").strip()
    if not pdf_url:
        pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"  # Default to a sample paper
        print(f"Using default PDF: {pdf_url}")
    
    # Example questions
    questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
    
    print("\nProcessing PDF and answering questions...")
    print("This may take a few minutes depending on the PDF size and model response time...")
    
    # Process PDF and get answers
    answers = rag.process_pdf_and_answer(pdf_url, questions, use_memory=True)
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for i, (question, answer) in enumerate(zip(questions, answers), 1):
        print( Fore.LIGHTMAGENTA_EX + f"\nQuestion {i}: {question}" + Fore.RESET)
        print(Fore.LIGHTCYAN_EX + f"Answer: {answer}" + Fore.RESET)
        print("-" * 80)


def demo_with_custom_questions():
    """
    Demo function allowing custom questions.
    """
    print("Initializing RAG system...")
    
    # Initialize RAG system
    rag = RAGSystem(llm_model=MODEL_NAME)
    
    # Get PDF URL
    pdf_url = input("Enter PDF URL: ").strip()
    if not pdf_url:
        print("No PDF URL provided. Exiting.")
        return
    
    # Create vector store
    success = rag.create_vector_store(pdf_url, use_memory=True)
    
    if not success:
        print("Failed to process PDF. Exiting.")
        return
    
    print("\nPDF processed successfully! You can now ask questions.")
    print("Type 'quit' to exit.")
    
    while True:
        question = input("\nEnter your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if question:
            answers = rag.answer_questions([question])
            print(f"\nAnswer: {answers[0]}")


if __name__ == "__main__":
    import sys
    import time
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        demo_with_custom_questions()
    else:
        t = time.time()
        main()
        print(f"\nTotal time taken: {time.time() - t:.2f} seconds")