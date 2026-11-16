
import os
from pathlib import Path

import chromadb
from mistralai.client import Mistral
from langchain_core.tools import tool


# ============================================================
# 1. PATHS & CHROMADB SETUP
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"

COLLECTION_NAME = "mancino_retail_rag_mistral"

_chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# ============================================================
# 2. MISTRAL CLIENT
# ============================================================

def get_mistral_client():

    api_key = os.environ.get("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY environment variable is not set."
        )

    return Mistral(api_key=api_key)


# ============================================================
# 3. GET COLLECTION
# ============================================================

def get_retail_collection():

    return _chroma_client.get_collection(
        name=COLLECTION_NAME
    )


# ============================================================
# 4. RETRIEVAL TOOL
# ============================================================

@tool
def search_retail_knowledge_base(
    query: str,
    top_k: int = 3
) -> str:

    """
    Search the retail knowledge base for relevant product,
    inventory, customer, transaction, sales, and promotion
    information using semantic search.
    """

    if not 1 <= top_k <= 20:
        return "Please provide top_k between 1 and 20."

    # Initialize clients
    mistral_client = get_mistral_client()
    collection = get_retail_collection()

    # Generate query embedding
    embed_response = mistral_client.embeddings.create(
        model="mistral-embed",
        inputs=[query],
    )

    query_embedding = embed_response.data[0].embedding

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ],
    )

    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []

    if not documents:
        return "No relevant documents found in the retail database."

    formatted_chunks = []

    for idx, (doc, meta, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):



        formatted_chunks.append(
            f"""
[Result {idx}]
Collection: {COLLECTION_NAME}
Metadata: {meta}
Distance: {distance}
Content: {doc}
"""
        )

    return "\n\n---\n\n".join(formatted_chunks)
