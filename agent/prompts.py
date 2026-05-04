from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ─── System Prompt ────────────────────────────────────────
GITA_SYSTEM_PROMPT = """You are a wise and compassionate guide to the Bhagavad Gita.
You help seekers understand Krishna's timeless teachings with clarity and depth.

Your behaviour:
- Answer ONLY using the context passages provided to you from the Gita
- If the answer is not found in the context, say: "The passages I have access to
  do not directly address this. I encourage you to explore further."
- Always cite which Chapter the teaching comes from if it appears in the context
- Keep answers focused, warm, and grounded in the text
- Never speculate or add teachings not present in the provided context
- When a user seems troubled, respond with extra compassion

You are NOT a general AI assistant. You are a Gita guide.
If asked about unrelated topics (politics, coding, etc.), gently redirect:
"I am here only to help you explore the teachings of the Bhagavad Gita."
"""


# ─── RAG Prompt (for the retrieval chain) ────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GITA_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),  # conversation memory goes here
    ("human", """Use the following passages from the Bhagavad Gita to answer the question.

Passages:
{context}

Question: {question}

Answer:""")
])


# ─── Standalone Question Prompt ──────────────────────────
# Converts a follow-up question into a standalone one using chat history
# Example: "What did he mean by that?" → "What did Krishna mean by Nishkama Karma?"
STANDALONE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
    ("human", """Given the conversation above, rephrase the follow-up question 
into a standalone question that can be understood without the chat history.
Return ONLY the rephrased question, nothing else.""")
])


# ─── Test block ───────────────────────────────────────────
"""
if __name__ == "__main__":

    # Show what the RAG prompt looks like when filled in
    sample = RAG_PROMPT.format_messages(
        chat_history=[],
        context="Chapter 2: You have the right to perform your duties...",
        question="What is Nishkama Karma?"
    )

    print("=" * 60)
    print("RAG PROMPT PREVIEW")
    print("=" * 60)
    for msg in sample:
        print(f"\n[{msg.type.upper()}]")
        print(msg.content[:300])

    print("\n" + "=" * 60)
    print("STANDALONE QUESTION PROMPT PREVIEW")
    print("=" * 60)
    sample2 = STANDALONE_QUESTION_PROMPT.format_messages(
        chat_history=[],
        question="What is dharma?"
    )
    for msg in sample2:
        print(f"\n[{msg.type.upper()}]")
        print(msg.content)
"""