import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from agent.prompts import RAG_PROMPT, STANDALONE_QUESTION_PROMPT
from agent.memory import wrap_with_memory, get_session_history
from knowledge_base.retriever import get_mmr_retriever

load_dotenv()


# ─── Initialise once — reused across all calls ────────────
_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,          # slight creativity, mostly grounded
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

_retriever = get_mmr_retriever(k=4)


# ─── Helper: format retrieved docs into clean string ──────
def _format_docs(docs: list[Document]) -> str:
    valid = [d for d in docs if len(d.page_content.strip()) > 30]
    if not valid:
        return "No relevant passages found in the Bhagavad Gita."
    return "\n\n".join(
        f"[Passage {i}]\n{d.page_content.strip()}"
        for i, d in enumerate(valid, 1)
    )


# ─── Build the chain ──────────────────────────────────────
def build_gita_chain():
    """
    Builds and returns the full RAG chain with memory.

    Flow:
    User question
        ↓
    Rephrase to standalone question (if follow-up)
        ↓
    Retrieve relevant Gita passages
        ↓
    Format context + question into prompt
        ↓
    LLM generates answer
        ↓
    Parse to string
    """

    # Step 1 — Rephrase follow-up questions into standalone ones
    # If chat_history is empty, just pass the question through unchanged
    def rephrase_if_needed(inputs: dict) -> str:
        history = inputs.get("chat_history", [])
        question = inputs["question"]

        if not history:
            return question  # first question — no rephrasing needed

        # Use LLM to rephrase follow-up into standalone question
        rephrased = STANDALONE_QUESTION_PROMPT | _llm | StrOutputParser()
        return rephrased.invoke({
            "chat_history": history,
            "question": question
        })

    # Step 2 — Retrieve docs based on (possibly rephrased) question
    def retrieve_context(standalone_question: str) -> str:
        docs = _retriever.invoke(standalone_question)
        return _format_docs(docs)

    # Step 3 — Assemble the full chain using LCEL (| pipe operator)
    chain = (
        RunnablePassthrough.assign(
            # Rephrase the question if it's a follow-up
            standalone_question=RunnableLambda(rephrase_if_needed),
        )
        | RunnablePassthrough.assign(
            # Retrieve Gita context using the (possibly rephrased) question
            context=RunnableLambda(lambda x: retrieve_context(x["standalone_question"]))
        )
        | RAG_PROMPT          # format into the prompt template
        | _llm                # send to LLM
        | StrOutputParser()   # extract plain string from response
    )

    # Step 4 — Wrap with memory (auto loads/saves chat history)
    chain_with_memory = wrap_with_memory(chain, input_key="question")

    return chain_with_memory


# ─── Public interface ─────────────────────────────────────
def ask_gita(question: str, session_id: str = "default") -> str:
    """
    Ask the Bhagavad Gita assistant a question.

    Args:
        question:   The user's question
        session_id: Unique ID per user/conversation (for memory isolation)

    Returns:
        The assistant's answer as a string
    """
    chain = build_gita_chain()
    response = chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )
    return response


"""
# ─── Test block ───────────────────────────────────────────
if __name__ == "__main__":

    SESSION = "yeshwant_test"

    print("🕉️  BHAGAVAD GITA AI ASSISTANT")
    print("=" * 60)

    # Turn 1 — Direct question
    q1 = "What does Krishna say about performing duty without attachment?"
    print(f"\n🧑 You: {q1}")
    a1 = ask_gita(q1, SESSION)
    print(f"\n🤖 Gita: {a1}")

    # Turn 2 — Follow-up (tests memory + rephrasing)
    q2 = "Can you give me a specific verse about this?"
    print(f"\n🧑 You: {q2}")
    a2 = ask_gita(q2, SESSION)
    print(f"\n🤖 Gita: {a2}")

    # Turn 3 — Another follow-up (deeper memory test)
    q3 = "How can I apply this teaching in daily life?"
    print(f"\n🧑 You: {q3}")
    a3 = ask_gita(q3, SESSION)
    print(f"\n🤖 Gita: {a3}")

    # Show what the agent remembers
    print("\n" + "=" * 60)
    print("📝 Conversation memory:")
    print("=" * 60)
    messages = get_session_history(SESSION).messages
    for msg in messages:
        role = "🧑" if msg.type == "human" else "🤖"
        print(f"{role} {msg.content[:80]}...")
"""