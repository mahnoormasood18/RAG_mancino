import os
from pathlib import Path
import chromadb
from mistralai.client import Mistral
from langchain_core.tools import tool

# ============================================================
# 1. CHROMADB HTTP CLIENT SETUP (Host/Port Connection)
# ============================================================

CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))

_chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT
)

DEFAULT_COLLECTION_NAME = "mancino_retail_rag_mistral"


# ============================================================
# 2. MISTRAL CLIENT
# ============================================================

def get_mistral_client():
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable is not set.")
    return Mistral(api_key=api_key)


# ============================================================
# 3. COLLECTION MANAGEMENT HELPER
# ============================================================

def get_retail_collection(collection_name: str = DEFAULT_COLLECTION_NAME):
    """Retrieves a specific Chroma collection by name."""
    return _chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


# ============================================================
# 4. RETRIEVAL TOOLS
# ============================================================

@tool
def search_retail_knowledge_base(
    query: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = 3
) -> str:
    """
    Search a specific Chroma collection in the retail database for relevant product,
    inventory, customer, transaction, sales, and promotion information using semantic search.
    """
    if not 1 <= top_k <= 20:
        return "Please provide top_k between 1 and 20."

    # Initialize clients & target collection
    mistral_client = get_mistral_client()
    collection = get_retail_collection(collection_name=collection_name)

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
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []

    if not documents:
        return f"No relevant documents found in collection '{collection_name}'."

    formatted_chunks = []
    for idx, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        formatted_chunks.append(
            f"""
[Result {idx}]
Collection: {collection_name}
Metadata: {meta}
Distance: {distance}
Content: {doc}
"""
        )

    return "\n\n---\n\n".join(formatted_chunks)


@tool
def get_collection(collection_name: str) -> str:
    """Fetch a specific ChromaDB collection by name. 
    Use this when you know which collection you need.
    """
    try:
        collection = _chroma_client.get_collection(name=colection_name)
        return (
            f"Collection found: {collection.name}\n"
            f"Total documents: {collection.count()}\n"
            f"Metadata: {collection.metadata}"
        )
    except Exception as e:
        return f"Collection '{collection_name}' not found. Error: {str(e)}"