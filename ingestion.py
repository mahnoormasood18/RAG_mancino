import json
import logging
import time
from pathlib import Path
import os  
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from mistralai.client import Mistral

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag_chatbot.ingestion")

# Define Custom Embedding Function for ChromaDB
class DirectMistralEmbeddingFunction(EmbeddingFunction):
    def __init__(self, client: Mistral, model: str = "mistral-embed"):
        self.client = client
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.embeddings.create(model=self.model, inputs=input)
        return [data.embedding for data in response.data]


def load_jsonl_documents(file_path: Path) -> tuple[list[str], list[str], list[dict]]:
    """Read JSONL documents file and extract documents, ids, and metadatas."""
    documents, ids, metadatas = [], [], []
    
    if not file_path.exists():
        raise FileNotFoundError(f"Combined documents file not found at: {file_path}")

    logger.info(f"Reading documents from {file_path.name}...")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            documents.append(doc["text"])
            ids.append(doc["id"])
            metadatas.append(doc["metadata"])

    return documents, ids, metadatas


def run_ingestion_pipeline(
    api_key: str,
    combined_file_path: Path,
    collection_name: str = "mancino_retail_rag_mistral",
    batch_size: int = 100,
    pause_between_batches: float = 1.0,
) -> None:
    """Orchestrates the vector database initialization via HTTP client and batch ingestion loop."""
    
    # 1. Initialize Mistral Client & Embedding Function
    mistral_client = Mistral(api_key=api_key)
    mistral_ef = DirectMistralEmbeddingFunction(
        client=mistral_client, model="mistral-embed"
    )

    # 2. Initialize ChromaDB HTTP Client (Host/Port)
    chroma_host = os.environ.get("CHROMA_HOST", "localhost")
    chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))
    
    logger.info(f"Connecting to ChromaDB server at {chroma_host}:{chroma_port}...")
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=mistral_ef,
        metadata={"hnsw:space": "cosine"},
    )

    # 3. Read data
    documents, ids, metadatas = load_jsonl_documents(combined_file_path)
    total_docs = len(documents)
    logger.info(f"Total documents to embed: {total_docs:,}")

    # 4. Ingestion Loop
    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        logger.info(f"Embedding batch {i:,} to {end_idx:,}...")

        success = False
        retries = 0
        while not success and retries < 5:
            try:
                collection.upsert(
                    documents=documents[i:end_idx],
                    ids=ids[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                )
                success = True
            except Exception as e:
                retries += 1
                wait_time = retries
                logger.warning(
                    f"Rate limit hit or error occurred: {e}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

        time.sleep(pause_between_batches)

    logger.info("Ingestion Complete using mistral-embed!")
    
    # Verification Peek
    sample_data = collection.peek(limit=5)
    logger.info(f"Total items in collection: {collection.count():,}")

    for idx, (doc_id, metadata) in enumerate(
        zip(sample_data["ids"], sample_data["metadatas"])
    ):
        logger.info(f"--- Record {idx+1} ---")
        logger.info(f"ID: {doc_id}")
        logger.info(f"Metadata: {metadata}")