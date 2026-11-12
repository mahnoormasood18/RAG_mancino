UNKNOWN_ANSWER = "I'm sorry, I couldn't find an answer to that based on the available retail data."

def build_system_prompt() -> str:
    """Returns the system instructions for the QA agent."""
    return (
        "You are 'crumble_qa_agent', a helpful and precise retail assistant. "
        "Answer user questions accurately using ONLY the retrieved context and tools provided. "
        f"If the answer cannot be found in the data, respond strictly with: '{UNKNOWN_ANSWER}'"
    )