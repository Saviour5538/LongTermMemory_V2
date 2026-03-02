# 🧠 Long‑Term Memory System (Mem0 Architecture)

A modular, AI‑powered long‑term memory system that extracts, updates, and retrieves user facts from conversations. Built with Groq LLMs, HuggingFace embeddings, and Qdrant vector database. Inspired by Mem0.

## Features

- **Conversation Fact Extraction** – Uses a Groq LLM (`llama-3.3-70b-versatile`) to extract personal facts about the user.
- **Intelligent Memory Update** – Decides whether to **ADD**, **UPDATE**, **DELETE**, or **NOOP** (no operation) each fact based on existing memory.
- **Vector Similarity Search** – Retrieves relevant memories using HuggingFace `all-MiniLM-L6-v2` embeddings stored in Qdrant.
- **Two Interfaces**:
  - `main.py` – CLI test runner with pre‑defined test conversations.
  - `app.py` – Interactive **Streamlit** chat application that injects memories into assistant responses.
- **Memory‑Aware Assistant** – In the Streamlit app, the assistant sees relevant memories before answering, enabling personalized conversations.

---

## Architecture

1. **Extractor** (`memory/extractor.py`)  
   - Takes a conversation (list of messages) and returns a list of factual statements about the user.
   - Uses a strict system prompt to ensure only current, third‑person facts are extracted.

2. **Updater** (`memory/updater.py`)  
   - For each extracted fact, retrieves the most similar existing memories (by vector similarity).
   - Sends the fact + similar memories to a Groq LLM, which decides the action:
     - **ADD** – new fact
     - **UPDATE** – refine an existing memory (preferred over DELETE)
     - **DELETE** – fact contradicts an old memory
     - **NOOP** – already present
   - Executes the corresponding operation on the Qdrant collection.

3. **Vector Store** (`memory/vector_store.py`)  
   - Manages a Qdrant collection (`long_term_memory`).
   - Generates embeddings via HuggingFace Inference API (`sentence-transformers/all-MiniLM-L6-v2`, 384‑dim).
   - Provides functions to add, update, delete, search, and retrieve all memories.

4. **Streamlit App** (`app.py`)  
   - Chat interface with memory pipeline visible in real‑time.
   - Before generating a response, it fetches memories relevant to the user’s message and injects them into the system prompt.
   - Shows step‑by‑step pipeline execution (extraction, retrieval, decision) in a side panel.

5. **CLI Test** (`main.py`)  
   - Runs a series of test conversations to demonstrate ADD, UPDATE/DELETE, and NOOP behaviour.
   - Prints extracted facts, decisions, and final memory state.

---

## Prerequisites

- Python 3.8+
- Accounts / API keys:
  - [Groq](https://groq.com) – for LLM (create an API key)
  - [HuggingFace](https://huggingface.co) – for embeddings (API key with access to inference)
  - [Qdrant Cloud](https://qdrant.tech) – free tier vector database (or run locally)

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Saviour5538/LongTermMemory_V2.git
   cd LongTermMemory_V2