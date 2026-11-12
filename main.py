import os
from pathlib import Path
import chromadb

from ingestion import run_ingestion_pipeline
from agent import agent_helper


def main() -> None:
    # 1. API Keys Configuration
    mistral_api_key = os.environ.get(
        "MISTRAL_API_KEY", "mstrl_1QyAlPfR9t8yyykQ2umCc5O8XTtDEP61_4gtV7V"
    )
    # Ensure Groq API key is set or remind the user
    if not os.environ.get("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY environment variable is not set!")
        print("Run: $env:GROQ_API_KEY=\"gsk_...\" in your terminal before running.")

    # 2. Paths (PROJECT_DIR points to RAG_project)
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    RAG_OUTPUT_DIR = PROJECT_DIR / "data" / "rag_documents"
    CHROMA_DIR = PROJECT_DIR / "chroma_db"
    COMBINED_FILE = RAG_OUTPUT_DIR / "all_retail_documents.jsonl"
    COLLECTION_NAME = "mancino_retail_rag_mistral"

    # 3. Check if vectors already exist before embedding
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        print(f"Collection is empty. Running ingestion from: {COMBINED_FILE}")
        run_ingestion_pipeline(
            api_key=mistral_api_key,
            combined_file_path=COMBINED_FILE,
            chroma_dir=CHROMA_DIR,
            collection_name=COLLECTION_NAME,
            batch_size=100,
            pause_between_batches=1.0,
        )
    else:
        print(
            f"Loaded existing vector store ({collection.count():,} documents). Skipping ingestion."
        )

    # 4. Start Interactive Agent Chat Loop
    print("\n--- Crumble QA Agent Ready (type 'exit' to quit) ---")
    session_id = "local_dev_session"

    while True:
        user_query = input("\nYou: ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        print("Agent: ", end="", flush=True)
        agent_helper(query=user_query, session_id=session_id)


if __name__ == "__main__":
    main()