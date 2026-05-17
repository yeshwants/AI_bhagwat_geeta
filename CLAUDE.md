# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bhagavad Gita AI Assistant — a RAG (Retrieval-Augmented Generation) conversational agent that answers questions grounded in the Bhagavad Gita text. Built with LangChain 0.3.x, ChromaDB, and OpenAI APIs.

## Commands

```bash
# Activate virtual environment (required before any command)
source venv/bin/activate

# Run the assistant (interactive terminal chat)
python main.py

# Build/rebuild the knowledge base (one-time, requires OPENAI_API_KEY)
# Delete gita_db/ first if rebuilding
python -m knowledge_base.embeddings

# Test individual modules
python -m knowledge_base.loader      # text loading + semantic chunking
python -m knowledge_base.retriever   # retrieval quality
python -m agent.prompts              # preview prompt templates
python -m agent.tools                # test retrieval tools
python -m agent.memory               # test session memory
python -m agent.agent                # full 3-turn conversation test
```

## Architecture

The app follows a RAG pipeline: **question -> rephrase -> retrieve -> prompt -> LLM -> answer**.

**`main.py`** — Terminal REPL. Handles user input, special commands (`clear`, `history`, `help`, `quit`), and calls `ask_gita()`.

**`agent/agent.py`** — Core LCEL chain assembly. Builds a 4-stage pipeline:
1. Rephrase follow-up questions into standalone queries (using `STANDALONE_QUESTION_PROMPT` + LLM, skipped if no chat history)
2. Retrieve top-4 diverse passages via MMR retriever
3. Format context + question into `RAG_PROMPT`
4. LLM (gpt-4o-mini, temp=0.3) generates answer

The chain is wrapped with `RunnableWithMessageHistory` for automatic session memory management. `build_gita_chain()` is called per invocation (constructs the chain fresh each time).

**`agent/prompts.py`** — Two prompt templates: `RAG_PROMPT` (system prompt + chat history + context + question) and `STANDALONE_QUESTION_PROMPT` (rewrites follow-ups into self-contained queries).

**`agent/memory.py`** — In-memory session store (`dict[str, ChatMessageHistory]`). Sessions are isolated by `session_id`. Memory is RAM-only, lost on restart.

**`agent/tools.py`** — Two LangChain `@tool` definitions: `search_gita` (general search) and `get_chapter_overview` (chapter-specific). These tools are defined but **not currently wired into the chain** — the chain uses the retriever directly.

**`knowledge_base/loader.py`** — Loads `data/gita.txt` via `TextLoader`, then splits using `SemanticChunker` (percentile threshold 90) which uses OpenAI embeddings to detect topic boundaries. Also has `load_gita()` for PDF loading.

**`knowledge_base/embeddings.py`** — Builds ChromaDB vector store from chunks using `text-embedding-3-small`. Persisted to `gita_db/`. Skips rebuild if `gita_db/` already has data.

**`knowledge_base/retriever.py`** — Provides `get_mmr_retriever(k)` (diverse results, fetch_k=10, lambda_mult=0.7) and `get_basic_retriever(k)` (simple similarity). The agent uses MMR.

## Key Details

- **LLM**: gpt-4o-mini via `langchain-openai`
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector store**: ChromaDB persisted to `gita_db/` (collection name: `bhagavad_gita`)
- **Source text**: `data/gita.txt` — 1909 public domain translation by Swami Swarupananda
- **Environment**: Requires `OPENAI_API_KEY` in `.env`
- **Python**: 3.13, uses `venv/`
- **LangChain version**: 0.3.x — import paths changed significantly in v1.x
