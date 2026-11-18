import os
from functools import lru_cache
from typing import Iterator

from mistralai.client import Mistral
from langchain_groq import ChatGroq  # <--- Use ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from tools import search_retail_knowledge_base, get_collection
from prompts import build_system_prompt, UNKNOWN_ANSWER

# 1. Mistral Client for query embeddings
_mistral_client = Mistral(
    api_key=os.environ.get("MISTRAL_API_KEY", "mstrl_1QyAlPfR9t8yyykQ2umCc5O8XTtDEP61_4gtV7V")
)

# 2. Build the LangGraph Agent with LRU Cache using Groq
@lru_cache(maxsize=1)
def _build_agent():
    chat_model = ChatGroq(
         model="openai/gpt-oss-120b",# Or your preferred Groq model
        temperature=0.1,
    )

    checkpointer = InMemorySaver()

    return create_agent(
        model=chat_model,
        tools=[search_retail_knowledge_base, get_collection],
        system_prompt=build_system_prompt(),
        name="crumble_qa_agent",
        checkpointer=checkpointer,
    )

# 3. Streaming & Helper Functions
def agent_helper_stream(query: str, session_id: str = "default") -> Iterator[str]:
    agent = _build_agent()
    inputs = {"messages": [{"role": "user", "content": query}]}
    config = {"configurable": {"thread_id": session_id}}

    for chunk, _metadata in agent.stream(inputs, config=config, stream_mode="messages"):
        if getattr(chunk, "type", "") == "AIMessageChunk" and chunk.content:
            yield chunk.content

def agent_helper(query: str, session_id: str = "default") -> str:
    answer_parts: list[str] = []

    for token in agent_helper_stream(query, session_id=session_id):
        print(token, end="", flush=True)
        answer_parts.append(token)

    print()
    answer = "".join(answer_parts).strip() or UNKNOWN_ANSWER
    return answer