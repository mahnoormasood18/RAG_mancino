import os
from pathlib import Path
from typing import Iterator

import chromadb
from mistralai.client import Mistral
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# 1. Paths & Vector Store Setup
PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "mancino_retail_rag_mistral"

# Initialize ChromaDB & Mistral client for query embeddings
_chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
_mistral_client = Mistral(
    api_key=os.environ.get("MISTRAL_API_KEY", "mstrl_1QyAlPfR9t8yyykQ2umCc5O8XTtDEP61_4gtV7V")
)


# 2. Define the ChromaDB Retrieval Tool
@tool
def search_retail_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the retail database for product details, inventory, promotions, policies, and customer info."""
    # Embed the incoming query with the exact same Mistral model used during ingestion
    embed_response = _mistral_client.embeddings.create(
        model="mistral-embed",
        inputs=[query],
    )
    query_embedding = embed_response.data[0].embedding

    # Query ChromaDB using the embedding vector
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return "No relevant documents found in the retail database."

    formatted_chunks = []
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        formatted_chunks.append(f"[Result {idx}] Metadata: {meta}\nContent: {doc}")

    return "\n\n---\n\n".join(formatted_chunks)


# 3. Build the LangGraph Agent with ChatGroq + Tool
def _build_agent():
    chat_model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1,
    )
    system_prompt = (
        "You are Crumble QA, a helpful retail operations assistant. "
        "Always use the `search_retail_knowledge_base` tool to look up factual product, "
        "inventory, promotion, or policy details before answering. "
        "If the retrieved records do not contain the answer, say so clearly without guessing."
    )

    checkpointer = InMemorySaver()

    return create_agent(
        model=chat_model,
        tools=[search_retail_knowledge_base],
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )


agent = _build_agent()


# 4. Streaming & Helper Functions
def agent_helper_stream(query: str, session_id: str = "default") -> Iterator[str]:
    inputs = {"messages": [{"role": "user", "content": query}]}
    config = {"configurable": {"thread_id": session_id}}

    for chunk, _metadata in agent.stream(inputs, config=config, stream_mode="messages"):
        # Stream only text tokens from the AI response (skip raw tool output chunks)
        if getattr(chunk, "type", "") == "AIMessageChunk" and chunk.content:
            yield chunk.content


def agent_helper(query: str, session_id: str = "default") -> str:
    full_response = []
    for token in agent_helper_stream(query, session_id=session_id):
        print(token, end="", flush=True)
        full_response.append(token)
    print()  # Newline at the end of the response
    return "".join(full_response)