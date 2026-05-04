# 🕉️ Bhagavad Gita AI Assistant

An intelligent conversational assistant built with **Agentic AI** and **Retrieval-Augmented Generation (RAG)** that answers questions grounded in the teachings of the Bhagavad Gita. Built as a step-by-step learning project to understand LangChain, ChromaDB, OpenAI embeddings, and agentic pipelines.

***

## ✨ Features

- 🔍 **Semantic Search** — finds the most meaningful Gita passages for any question
- 🧠 **Conversation Memory** — remembers context across multiple turns in a session
- 🤖 **Agentic Pipeline** — uses LCEL (LangChain Expression Language) to reason and retrieve
- 📚 **RAG-Grounded Answers** — responses are anchored to actual Gita text, not hallucinated
- 🔄 **Follow-up Understanding** — rephrases vague follow-ups ("explain that") into searchable queries
- 🌿 **Semantic Chunking** — uses OpenAI embeddings to split text by meaning, not character count
- 💾 **Persistent Vector DB** — ChromaDB stores embeddings on disk; no re-embedding on restart

***

## 🗂️ Project Structure

```
geeta_ai_assistant/
│
├── 📄 .env                          ← API keys (never commit this)
├── 📄 requirements.txt              ← All dependencies
├── 📄 main.py                       ← Entry point — run this to chat
│
├── 📁 data/
│   ├── download_gita.py             ← Scrapes Gita text from sacred-texts.com
│   └── gita.txt                     ← Downloaded Gita source (auto-generated)
│
├── 📁 knowledge_base/
│   ├── __init__.py
│   ├── loader.py                    ← Loads & semantically chunks gita.txt
│   ├── embeddings.py                ← Embeds chunks → stores in ChromaDB
│   └── retriever.py                 ← MMR retriever over the vector store
│
├── 📁 gita_db/                      ← ChromaDB vectors (auto-generated, ~131 vectors)
│
├── 📁 agent/
│   ├── __init__.py
│   ├── prompts.py                   ← System prompt, RAG prompt, standalone Q prompt
│   ├── tools.py                     ← @tool definitions for the agent
│   ├── memory.py                    ← Session-based conversation memory
│   └── agent.py                     ← LCEL chain assembly + ask_gita() function
│
└── 📁 utils/
    ├── __init__.py
    └── helpers.py                   ← Utility functions
```

***

## 🚀 Setup & Installation

### 1. Clone or create the project

```bash
mkdir geeta_ai_assistant
cd geeta_ai_assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Download the Gita text

```bash
python -m data.download_gita
```

This scrapes all 18 chapters from [sacred-texts.com](https://www.sacred-texts.com/hin/sbg/index.htm) (public domain, 1909 translation by Swami Swarupananda) and saves them as `data/gita.txt`.

### 5. Build the knowledge base (run once)

```bash
python -m knowledge_base.embeddings
```

This uses OpenAI's `text-embedding-3-small` model to semantically chunk and embed the entire Gita into ChromaDB. Costs ~$0.01 and takes 1–2 minutes. **Only needs to run once** — vectors are saved to `gita_db/` on disk.

### 6. Start the assistant

```bash
python main.py
```

***

## 💬 Usage

```
╔══════════════════════════════════════════════════════════╗
║         🕉️  BHAGAVAD GITA AI ASSISTANT  🕉️              ║
╚══════════════════════════════════════════════════════════╝

🧑 You   : What does Krishna say about performing duty without attachment?
🤖 Gita  : Krishna teaches the principle of Nishkama Karma — performing
            one's duty without attachment to results...
```

### Special Commands

| Command   | Action |
|-----------|--------|
| `clear`   | Start a fresh conversation (clears memory) |
| `history` | View the full conversation so far |
| `help`    | Show available commands |
| `quit`    | Exit the assistant |

***

## 🧩 How It Works

### The RAG Pipeline

```
User Question
      ↓
Rephrase to standalone question   ← uses chat_history + LLM
      ↓
MMR Retrieval from ChromaDB       ← top-4 diverse Gita passages
      ↓
Format context + question         ← fills RAG_PROMPT template
      ↓
GPT-4o-mini generates answer      ← grounded in retrieved passages
      ↓
Save turn to session memory       ← for future follow-ups
      ↓
Answer returned to user
```

### Chunking Strategy

The Gita text is split using **Semantic Chunking** (via `langchain-experimental`). Unlike fixed-size chunking, semantic chunking uses OpenAI embeddings to detect where topics change — ensuring each shloka or teaching stays in one cohesive chunk.

```
Fixed chunking (500 chars)  →  may split a shloka mid-sentence ❌
Semantic chunking           →  splits only when meaning changes  ✅
```

### Retrieval Strategy

**MMR (Maximal Marginal Relevance)** retrieval is used instead of basic similarity search:

- Fetches 10 candidate chunks
- Returns the 4 most **relevant AND diverse** results
- Avoids returning 4 near-identical passages about the same verse

### Memory Architecture

Each session gets an isolated `ChatMessageHistory` stored in RAM. The `STANDALONE_QUESTION_PROMPT` converts follow-up questions into self-contained queries before retrieval:

```
Turn 1: "What is Nishkama Karma?"
Turn 2: "Can you explain that more simply?"
         ↓ rephrased to →
        "Can you explain Nishkama Karma more simply?"
```

***

## 📦 Dependencies

```txt
langchain==0.3.25
langchain-community==0.3.23
langchain-core==0.3.59
langchain-openai
langchain-experimental
langchain-chroma
langchain-text-splitters
chromadb
sentence-transformers
pypdf
python-dotenv
openai
beautifulsoup4
requests
```

> **Note:** This project targets **LangChain 0.3.x**. LangChain v1.x restructured import paths significantly — see the [v1 migration guide](https://docs.langchain.com/oss/python/migrate/langchain-v1) if upgrading.

***

## 🔧 Individual Module Testing

Each module can be tested independently:

```bash
# Test text loading and chunking
python -m knowledge_base.loader

# Build or verify knowledge base
python -m knowledge_base.embeddings

# Test retrieval quality
python -m knowledge_base.retriever

# Preview prompt templates
python -m agent.prompts

# Test agent tools
python -m agent.tools

# Test memory management
python -m agent.memory

# Full 3-turn conversation test
python -m agent.agent
```

***

## 💰 Cost Estimate

| Operation | Model | Approx Cost |
|-----------|-------|-------------|
| Build knowledge base (once) | text-embedding-3-small | ~$0.01 |
| Embed each user question | text-embedding-3-small | ~$0.000002 |
| Generate each answer | gpt-4o-mini | ~$0.0002 |
| **1,000 user questions** | Combined | **~$0.22** |

***

## 🗺️ Learning Concepts Covered

| Concept | File |
|---------|------|
| Document loading & text splitting | `knowledge_base/loader.py` |
| Semantic chunking with embeddings | `knowledge_base/loader.py` |
| Vector embeddings & ChromaDB | `knowledge_base/embeddings.py` |
| MMR retrieval | `knowledge_base/retriever.py` |
| Prompt engineering | `agent/prompts.py` |
| LangChain `@tool` decorator | `agent/tools.py` |
| Session-based memory | `agent/memory.py` |
| LCEL pipeline (`\|` operator) | `agent/agent.py` |
| `RunnableWithMessageHistory` | `agent/agent.py` |
| Terminal chat interface | `main.py` |

***

## 🚀 Next Steps & Ideas

- **Hindi mode** — change system prompt to respond in Hindi
- **Streamlit UI** — add a web interface in ~10 lines of code
- **Persistent memory** — swap `ChatMessageHistory` for `SQLChatMessageHistory`
- **Chapter navigation** — add a tool to fetch all verses from a specific chapter
- **Multiple texts** — add Upanishads or Yoga Sutras as additional collections
- **Evaluation layer** — score answer faithfulness against retrieved context
- **Deploy** — host on HuggingFace Spaces for free

***

## 📖 Source Text

The Gita text is sourced from [sacred-texts.com](https://www.sacred-texts.com/hin/sbg/index.htm) — the 1909 English translation by **Swami Swarupananda**, which is in the **public domain**.

***

## 🙏 Acknowledgements

Built as a learning project exploring:
- [IBM's Agentic AI Architecture Guide](https://www.ibm.com/think/architectures/patterns/agentic-ai)
- [IBM's RAG Cookbook](https://www.ibm.com/think/architectures/rag-cookbook)
- [LangChain Documentation](https://docs.langchain.com)

> *"You have a right to perform your prescribed duties, but you are not entitled to the fruits of your actions."* — Bhagavad Gita, Chapter 2, Verse 47