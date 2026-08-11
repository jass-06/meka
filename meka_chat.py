"""
meka_chat.py

Quick terminal chat loop against the same RAGPipeline used by app.py and
api.py - handy for smoke-testing ingestion/retrieval/LLM routing without
spinning up Streamlit.
"""
from rag.retrieval import RAGPipeline

if __name__ == "__main__":
    print("MEKA - Multimodal Enterprise Knowledge Assistant (terminal mode)")
    print("Type 'exit' to quit.\n")

    pipeline = RAGPipeline()

    if pipeline.is_empty():
        print("No documents found. Run `python offline_ingest.py` first.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        result = pipeline.process_question(q)
        if result.get("error"):
            print(f"\nError: {result['error']}\n")
            continue

        print(f"\nMEKA ({result.get('used_llm', 'none')}): {result['answer']}")
        if result.get("sources"):
            print(f"Sources: {', '.join(result['sources'])}")
        print()
