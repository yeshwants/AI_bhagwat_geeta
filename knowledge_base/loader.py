import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings


load_dotenv()

def load_gita(pdf_path: str):
    """
    Loads the Bhagavad Gita PDF and returns raw documents.
    """
    print(f"📖 Loading document from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")
    return documents

def load_gita2(file_path: str):
    print(f"📖 Loading document from: {file_path}")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents")
    return documents


def chunk_documents(documents):
    """
    Splits documents into semantic chunks using OpenAI embeddings.
    Each chunk will contain one complete, meaningful thought/shloka.
    """

    # OpenAI's embedding model - converts sentences to vectors for comparison
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # SemanticChunker splits based on MEANING, not character count
    splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )

    print("🔍 Running semantic chunking (this may take a moment...)")
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} semantic chunks")
    return chunks

"""
if __name__ == "__main__":
    docs = load_gita2("data/gita.txt")
    chunks = chunk_documents(docs)

    print(f"\n📊 Chunk size stats:")
    lengths = [len(c.page_content) for c in chunks]
    print(f"   Shortest chunk : {min(lengths)} characters")
    print(f"   Longest chunk  : {max(lengths)} characters")
    print(f"   Average chunk  : {int(sum(lengths)/len(lengths))} characters")

    print(f"\n--- Sample Chunk 1 ---")
    print(chunks[0].page_content)

    print(f"\n--- Sample Chunk 2 ---")
    print(chunks[1].page_content)
"""