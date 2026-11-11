import os
from pathlib import Path
from ingestion import run_ingestion_pipeline

def main() -> None:
    # Set API Key securely
    api_key = os.environ.get("MISTRAL_API_KEY", "mstrl_1QyAlPfR9t8yyykQ2umCc5O8XTtDEP61_4gtV7V")
    
    # PROJECT_DIR points to RAG_project (going up two levels from working/)
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    RAG_OUTPUT_DIR = PROJECT_DIR / "data" / "rag_documents"
    CHROMA_DIR = PROJECT_DIR / "chroma_db"
    COMBINED_FILE = RAG_OUTPUT_DIR / "all_retail_documents.jsonl"

    print(f"Looking for data at: {COMBINED_FILE}")

    # Run the pipeline function defined in ingestion.py
    run_ingestion_pipeline(
        api_key=api_key,
        combined_file_path=COMBINED_FILE,
        chroma_dir=CHROMA_DIR,
        collection_name="mancino_retail_rag_mistral",
        batch_size=100,
        pause_between_batches=1.0
    )

if __name__ == "__main__":
    main()