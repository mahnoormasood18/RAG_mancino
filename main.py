import os
from pathlib import Path
import chromadb

from ingestion import run_ingestion_pipeline
from agent import agent_helper


def main() -> None:
    # 1. API Keys & Configuration
    mistral_api_key = os.environ["MISTRAL_API_KEY"]

    chroma_host = os.environ.get("CHROMA_HOST", "localhost")
    chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))

    # 2. Paths
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    RAG_OUTPUT_DIR = PROJECT_DIR / "data" / "rag_documents"
    COMBINED_FILE = RAG_OUTPUT_DIR / "all_retail_documents.jsonl"
    COLLECTION_NAME = "mancino_retail_rag_mistral"

    # 3. Connect to ChromaDB
    client = chromadb.HttpClient(
        host=chroma_host,
        port=chroma_port
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 4. Check whether ingestion is required
    if collection.count() == 0:
        print(f"Collection is empty. Running ingestion from: {COMBINED_FILE}")

        run_ingestion_pipeline(
            api_key=mistral_api_key,
            combined_file_path=COMBINED_FILE,
            collection_name=COLLECTION_NAME,
            batch_size=100,
            pause_between_batches=1.0,
        )
    else:
        print(
            f"Loaded existing vector store collection "
            f"'{COLLECTION_NAME}' ({collection.count():,} documents). "
            "Skipping ingestion."
        )

    # 5. Start Interactive Agent Chat Loop
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

        agent_helper(
            query=user_query,
            session_id=session_id
        )


if __name__ == "__main__":
    main()