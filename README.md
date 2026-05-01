# 🤖 Deep Learning RAG Interview Prep Agent

A RAG (Retrieval-Augmented Generation) system I built using LangChain, LangGraph, and ChromaDB. Upload deep learning study material, then chat with an AI agent that retrieves relevant content and generates technical interview questions — with source citations and hallucination guards.

**Live Demo:** https://rag-agent-ccmufe7hlptmlcrvlsjin4.streamlit.app/

---

## 🧠 What I Built

I wanted to go deeper than basic LLM API calls and actually build a production-style RAG pipeline with proper chunking, vector storage, duplicate detection, and a stateful multi-node agent. This project uses LangGraph to wire together retrieval, question generation, and answer evaluation as distinct nodes in a graph — the same pattern used in real AI systems.

---

## 🏗️ System Architecture

```
User uploads study material (PDF / Markdown)
        ↓
[CHUNKER]  Split into atomic topic chunks with metadata
        ↓
[VECTORSTORE]  Embed + store in ChromaDB (with duplicate detection)
        ↓
User asks a question
        ↓
[LANGGRAPH AGENT]
  Node 1: Retrieve relevant chunks from ChromaDB
  Node 2: Generate interview question from context
  Node 3: Evaluate user's answer against source material
        ↓
Response with source citations + hallucination guard
```

---

## ✨ Key Features

- **Document Ingestion** — Upload PDFs and Markdown files; chunks stored with rich metadata (topic, difficulty, type, source)
- **Duplicate Detection** — Prevents re-ingesting the same content into the vector store
- **RAG Pipeline** — ChromaDB retrieval feeds LangGraph nodes for grounded, cited responses
- **Question Generation** — Generates interview questions at configurable difficulty levels (beginner / intermediate / advanced)
- **Answer Evaluation** — Scores candidate answers out of 10 with detailed feedback against source chunks
- **Hallucination Guard** — Agent explicitly signals when no relevant context is found rather than making things up
- **Source Citations** — Every response cites the exact chunk and source file it drew from
- **Streamlit UI** — Three-panel interface: document ingestion, corpus viewer, chat

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Agent Framework | LangChain + LangGraph |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace (local) or OpenAI |
| LLM Backend | Groq (Llama 3) / Ollama / LM Studio |
| UI | Streamlit |
| Package Manager | UV |

---

## 📂 Project Structure

```
rag-agent/
├── src/rag_agent/
│   ├── config.py              # LLM + embedding factory, settings
│   ├── corpus/
│   │   └── chunker.py         # Document chunking with metadata
│   ├── vectorstore/
│   │   └── store.py           # ChromaDB manager + duplicate detection
│   ├── agent/
│   │   ├── state.py           # AgentState + data models
│   │   ├── prompts.py         # All LLM prompt templates
│   │   ├── nodes.py           # LangGraph node functions
│   │   └── graph.py           # Graph assembly
│   └── ui/
│       └── app.py             # Streamlit application
├── data/corpus/               # Study material (PDF + Markdown)
├── tests/
│   └── test_vectorstore.py    # Unit tests
├── examples/
│   └── sample_chunk.json      # Canonical chunk schema reference
├── docs/
│   └── architecture.md        # System design notes
├── .env.example
├── pyproject.toml
└── README.md
```

---

## ⚙️ How to Run

```bash
git clone https://github.com/Karthik2920/rag-agent.git
cd rag-agent

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Add your LLM provider key (Groq recommended — free tier available)

# Launch the app
uv run streamlit run src/rag_agent/ui/app.py
```

### LLM Provider Options

| Provider | Setup |
|---|---|
| **Groq** (recommended) | Free API key at console.groq.com → set `LLM_PROVIDER=groq` |
| **Ollama** (fully local) | `ollama pull llama3.2` → set `LLM_PROVIDER=ollama` |
| **LM Studio** | Load model → Start local server → set `LLM_PROVIDER=lmstudio` |

---

## 🧪 Running Tests

```bash
uv run pytest tests/ -v
```

---

## 📚 Corpus Coverage

The included study corpus covers 10 deep learning topics:
ANN, CNN, RNN, LSTM, Seq2Seq, Autoencoder, GAN, Boltzmann Machine, SOM — with landmark papers (LeCun, Hochreiter & Schmidhuber, Goodfellow et al.) as source PDFs.

---

## 🙋 About

Built by **Karthik Saraf** to get hands-on experience with RAG system design, LangGraph agent orchestration, and vector database management — skills directly applicable to AI/ML engineering and data science roles.
