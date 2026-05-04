from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# In-memory store: session_id → ChatMessageHistory
# Each user session gets its own isolated conversation history
_session_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Returns the chat history for a given session.
    Creates a new one if it doesn't exist yet.
    """
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def clear_session(session_id: str) -> None:
    """
    Clears the conversation history for a session.
    Useful when user wants to start fresh.
    """
    if session_id in _session_store:
        _session_store[session_id].clear()
        print(f"🧹 Session '{session_id}' cleared.")
    else:
        print(f"ℹ️  No session found for '{session_id}'.")


def get_session_messages(session_id: str) -> list:
    """
    Returns all messages in a session as a list.
    Useful for debugging what the agent remembers.
    """
    history = get_session_history(session_id)
    return history.messages


def wrap_with_memory(chain, input_key: str = "question"):
    """
    Wraps any LangChain chain/runnable with message history.
    The returned chain automatically loads and saves chat history.
    """
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key=input_key,
        history_messages_key="chat_history"
    )

"""
# ─── Test block ───────────────────────────────────────────
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, AIMessage

    SESSION = "test_user_1"

    # Simulate a conversation being stored
    history = get_session_history(SESSION)
    history.add_message(HumanMessage(content="What is dharma?"))
    history.add_message(AIMessage(content="Dharma is one's sacred duty..."))
    history.add_message(HumanMessage(content="Can you give an example?"))
    history.add_message(AIMessage(content="Arjuna's dharma as a warrior..."))

    print("=" * 60)
    print(f"Session: '{SESSION}' — {len(get_session_messages(SESSION))} messages")
    print("=" * 60)

    for msg in get_session_messages(SESSION):
        role = "🧑 User " if isinstance(msg, HumanMessage) else "🤖 Gita "
        print(f"\n{role}: {msg.content[:80]}")

    print("\n" + "=" * 60)
    print("Clearing session...")
    print("=" * 60)
    clear_session(SESSION)
    print(f"Messages after clear: {len(get_session_messages(SESSION))}")
"""