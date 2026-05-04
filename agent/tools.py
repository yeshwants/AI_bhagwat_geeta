from langchain_core.tools import tool
from knowledge_base.retriever import get_mmr_retriever


# Initialise retriever once — reused across all tool calls
retriever = get_mmr_retriever(k=4)


def _format_docs(docs) -> str:
    """
    Filters empty chunks and formats retrieved docs
    into a single clean string for the LLM.
    """
    valid_docs = [doc for doc in docs if len(doc.page_content.strip()) > 30]

    if not valid_docs:
        return "No relevant passages found."

    formatted = []
    for i, doc in enumerate(valid_docs, 1):
        source = doc.metadata.get("source", "Bhagavad Gita")
        formatted.append(f"[Passage {i}]\n{doc.page_content.strip()}")

    return "\n\n".join(formatted)


@tool
def search_gita(query: str) -> str:
    """
    Search the Bhagavad Gita knowledge base for teachings, shlokas,
    and philosophical guidance related to the query.
    Use this for ANY question about the Gita, its chapters,
    verses, characters, or teachings.
    """
    docs = retriever.invoke(query)
    return _format_docs(docs)


@tool
def get_chapter_overview(chapter_number: int) -> str:
    """
    Get an overview of a specific chapter of the Bhagavad Gita.
    Use this when the user asks about what a particular chapter covers.
    Input must be a chapter number between 1 and 18.
    """
    if not 1 <= chapter_number <= 18:
        return "Invalid chapter number. The Bhagavad Gita has 18 chapters (1-18)."

    query = f"Chapter {chapter_number} Bhagavad Gita summary overview"
    docs = retriever.invoke(query)
    return _format_docs(docs)


# List of all tools — imported by agent.py
GITA_TOOLS = [search_gita, get_chapter_overview]

"""
# ─── Test block ───────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1 — search_gita tool")
    print("=" * 60)
    result1 = search_gita.invoke("What is the nature of the eternal soul?")
    print(result1)

    print("\n" + "=" * 60)
    print("TEST 2 — get_chapter_overview tool")
    print("=" * 60)
    result2 = get_chapter_overview.invoke({"chapter_number": 2})
    print(result2)

    print("\n" + "=" * 60)
    print("TEST 3 — Tool metadata (what the agent sees)")
    print("=" * 60)
    for t in GITA_TOOLS:
        print(f"\nTool name : {t.name}")
        print(f"Description: {t.description}")
"""