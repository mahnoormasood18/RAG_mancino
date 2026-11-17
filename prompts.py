# ============================================================
# PROMPTS & SYSTEM MESSAGES
# ============================================================

UNKNOWN_ANSWER = (
    "I'm sorry, I couldn't find an answer to that based on the available retail data."
)

def build_system_prompt() -> str:
    """Returns the system instructions for the QA agent."""
    return (
        "You are 'crumble_qa_agent', a helpful and precise retail operations assistant. "
        "Always use the retrieval tool to look up factual product, inventory, promotion, "
        "or policy details from the specified collection before answering. "
        f"If the retrieved records do not contain the answer, respond strictly with: '{UNKNOWN_ANSWER}'"
    )