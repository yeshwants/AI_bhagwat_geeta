import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

CHROMA_PATH = "./gita_db"
COLLECTION_NAME = "bhagavad_gita"


def get_vectorstore():
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )


def get_basic_retriever(k: int = 3):
    """Top-k most similar chunks. Fast and cheap."""
    retriever = get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    print(f"✅ Basic retriever ready (top-{k} results)")
    return retriever


def get_mmr_retriever(k: int = 3):
    """Diverse results — avoids returning near-duplicate chunks."""
    retriever = get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 10, "lambda_mult": 0.7}
    )
    print(f"✅ MMR retriever ready (diverse top-{k} results)")
    return retriever


# ─── Test block ───────────────────────────────────────────
"""
if __name__ == "__main__":

    query = "What does Krishna say about performing duty without attachment?"

    print("=" * 60)
    print("TEST 1 — Basic Retriever")
    print("=" * 60)
    r1 = get_basic_retriever(k=3)
    for i, doc in enumerate(r1.invoke(query), 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:250])

    print("\n" + "=" * 60)
    print("TEST 2 — MMR Retriever")
    print("=" * 60)
    r2 = get_mmr_retriever(k=3)
    for i, doc in enumerate(r2.invoke(query), 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:250])
"""