import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from knowledge_base.loader import chunk_documents, load_gita2

load_dotenv()

CHROMA_PATH = "./gita_db"
COLLECTION_NAME = "bhagavad_gita"


def get_embedding_model():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )


def knowledge_base_exists() -> bool:
    """
    Returns True only if gita_db/ exists AND has actual vector data in it.
    An empty folder returns False.
    """
    return os.path.exists(CHROMA_PATH) and bool(os.listdir(CHROMA_PATH))


def build_knowledge_base(file_path: str):
    """
    Loads the Gita, chunks it, embeds it, and saves to ChromaDB.
    Run this ONCE to build the knowledge base.
    """
    # Step 1 — Load and chunk
    documents = load_gita2(file_path)
    chunks = chunk_documents(documents)

    # Step 2 — Prepare embedding model
    embedding_model = get_embedding_model()

    # ✅ Fixed: use knowledge_base_exists() not just os.path.exists()
    if knowledge_base_exists():
        print(f"⚠️  Knowledge base already exists at '{CHROMA_PATH}'")
        print("    Delete the 'gita_db/' folder to rebuild it.")
        return load_knowledge_base()

    # Step 3 — Build and store
    print(f"\n🔢 Creating embeddings and storing in ChromaDB...")
    print(f"   This calls OpenAI API once — may take 1-2 minutes...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )

    count = vectorstore._collection.count()
    print(f"✅ Knowledge base built successfully!")
    print(f"   Total vectors stored: {count}")
    print(f"   Saved to: {CHROMA_PATH}/")

    return vectorstore


def load_knowledge_base():
    """
    Loads an already-built knowledge base from disk.
    No API calls — reads directly from local disk.
    """
    print(f"📂 Loading existing knowledge base from '{CHROMA_PATH}'...")

    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )

    count = vectorstore._collection.count()
    print(f"✅ Knowledge base loaded — {count} vectors ready")
    return vectorstore


# ─── Test block ───────────────────────────────────────────
"""
if __name__ == "__main__":

    if knowledge_base_exists():
        print("🔍 Knowledge base found — loading it...")
        vs = load_knowledge_base()
    else:
        print("🏗️  No knowledge base found — building it now...")
        vs = build_knowledge_base("data/gita.txt")

    # Sanity test
    print("\n🧪 Running a test search...")
    test_query = "What is the duty of a warrior?"
    results = vs.similarity_search(test_query, k=3)

    print(f"\nQuery: '{test_query}'")
    print(f"Top {len(results)} results:\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(doc.page_content[:200])
        print()
"""