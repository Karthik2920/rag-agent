"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager
from rag_agent.agent.prompts import SYSTEM_PROMPT
from rag_agent.agent.prompts import SYSTEM_PROMPT, ANSWER_PROMPT
from rag_agent.agent.graph import get_compiled_graph

# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": "default-session",  # LangGraph conversation thread
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of PDF and Markdown files. Displays
    ingestion results (chunks added, duplicates skipped, errors).
    Updates the ingested documents list after successful ingestion.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """
    st.sidebar.header("📂 Corpus Ingestion")
    uploaded_files = st.sidebar.file_uploader(
    "Upload study materials",
    type=["pdf", "md"],
    accept_multiple_files=True
)
    if uploaded_files:
        if st.sidebar.button("Ingest Documents"):
            import tempfile
            from pathlib import Path

            file_paths = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                    tmp.write(file.getvalue())
                file_paths.append((Path(tmp.name), file.name))

            progress = st.sidebar.progress(0, text="Chunking files...")
            chunks = chunker.chunk_files(file_paths)
            progress.progress(50, text="Embedding & storing...")
            result = store.ingest(chunks)
            progress.progress(100, text="Done!")
            progress.empty()

            st.sidebar.success(
                f"✅ {result.ingested} chunks added, {result.skipped} duplicates skipped"
            )
            if result.errors:
                st.sidebar.error(f"Errors: {result.errors}")

    # TODO: implement
    # 1. st.sidebar.file_uploader(
    #        "Upload study materials",
    #        type=["pdf", "md"],
    #        accept_multiple_files=True
    #    )
    #
    # 2. "Ingest Documents" button — only enabled when files are selected
    #
    # 3. On button click:
    #    a. Save uploaded files to a temp directory
    #    b. chunker.chunk_files(file_paths)
    #    c. store.ingest(chunks) → IngestionResult
    #    d. Display result: st.success / st.warning / st.error
    #       Show: "{result.ingested} chunks added, {result.skipped} duplicates skipped"
    #    e. Refresh ingested documents list in session_state
    #
    # 4. Render ingested documents list below the uploader
    #    For each document: show source name, topic, chunk count
    #    Add a small "🗑 Remove" button per document that calls store.delete_document()

    # st.sidebar.info("Upload .pdf or .md files to populate the corpus.")


def render_corpus_stats(store: VectorStoreManager) -> None:
    try:
        count = store._collection.count()
        if count == 0:
            return
        all_meta = store._collection.get(include=["metadatas"])
        topics = sorted({m.get("topic", "?") for m in all_meta["metadatas"]})
        st.sidebar.divider()
        st.sidebar.markdown(
            f"📊 **{len(topics)} topics · {count} chunks**  \n"
            + "  ".join([f"`{t}`" for t in topics])
        )
    except:
        pass


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


# def render_document_viewer(store: VectorStoreManager) -> None:
def render_document_viewer(store: VectorStoreManager) -> None:
    st.subheader("📄 Document Viewer")

    try:
        docs = store._collection.get(include=["metadatas", "documents"])
    except:
        st.info("No documents found.")
        return

    if not docs or not docs.get("documents"):
        st.info("No documents ingested yet.")
        return

    documents = docs["documents"]
    metadatas = docs["metadatas"]

    # Group chunks by source
    from collections import defaultdict
    grouped = defaultdict(list)
    for text, meta in zip(documents, metadatas):
        source = meta.get("source", "Unknown")
        grouped[source].append((text, meta))

    total_chunks = len(documents)
    total_docs = len(grouped)
    st.caption(f"{total_docs} documents · {total_chunks} total chunks")
    st.divider()

    for source, chunk_list in grouped.items():
        topic = chunk_list[0][1].get("topic", "?")
        difficulty = chunk_list[0][1].get("difficulty", "?")
        with st.expander(f"📄 {source}  ·  {len(chunk_list)} chunks  ·  [{topic} | {difficulty}]"):
            for i, (text, meta) in enumerate(chunk_list):
                st.markdown(f"**Chunk {i+1}**")
                st.text(text[:400] + ("..." if len(text) > 400 else ""))
                st.divider()


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)

def _is_insufficient(text: str) -> bool:
    return any(phrase in text.lower() for phrase in [
        "does not contain enough information",
        "not enough information",
        "cannot answer",
        "no relevant",
        "does not mention",
    ])


def render_chat_panel(graph):
    from langchain_core.messages import HumanMessage

    # Header row with Clear Chat button
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.subheader("💬 Interview Prep Chat")
        st.caption("Ask anything about ANN, CNN, RNN, LSTM, Seq2Seq, or Autoencoder — answers are grounded in your uploaded documents.")
    with col_btn:
        st.write("")
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Welcome message when chat is empty
    if not st.session_state.chat_history:
        st.markdown("**Try one of these questions to get started:**")
        example_questions = [
            "What is backpropagation?",
            "How do CNNs detect features in images?",
            "What is the vanishing gradient problem?",
            "How does an LSTM remember long-term information?",
        ]
        col1, col2 = st.columns(2)
        for i, q in enumerate(example_questions):
            col = col1 if i % 2 == 0 else col2
            if col.button(q, use_container_width=True):
                st.session_state["_pending_query"] = q
                st.rerun()

    # Replay chat history with timestamps
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("timestamp"):
                st.caption(msg["timestamp"])
            if msg["role"] == "assistant":
                insufficient = _is_insufficient(msg["content"])
                if msg.get("sources") and not msg.get("no_context_found") and not insufficient:
                    with st.expander("📚 Sources"):
                        for src in msg["sources"]:
                            st.markdown(f"- 📄 **{src}**")

    query = st.chat_input("Type your question here...")

    # Handle example question button clicks
    if "_pending_query" in st.session_state:
        query = st.session_state.pop("_pending_query")

    if query:
        st.chat_message("user").write(query)

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        with st.spinner("Thinking..."):
            result = graph.invoke(
                {"messages": [HumanMessage(content=query)]},
                config=config,
            )

        final = result.get("final_response")
        if final:
            answer = final.answer
            sources = list(dict.fromkeys(final.sources))
            no_context = final.no_context_found
            confidence = final.confidence
            rewritten_query = final.rewritten_query
        else:
            answer = "Something went wrong — no response generated."
            sources = []
            no_context = False
            confidence = None
            rewritten_query = None

        insufficient = _is_insufficient(answer)
        with st.chat_message("assistant"):
            st.write(answer)
            if sources and not no_context and not insufficient:
                with st.expander("📚 Sources"):
                    for src in sources:
                        st.markdown(f"- 📄 **{src}**")
            if no_context or insufficient:
                st.warning("⚠️ No relevant content found in corpus.")

        from datetime import datetime
        ts = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append({"role": "user", "content": query, "timestamp": ts})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "no_context_found": no_context,
            "confidence": confidence,
            "rewritten_query": rewritten_query,
            "original_query": query,
            "timestamp": ts,
        })
        st.rerun()
# ---------------------------------------------------------------------------


def render_chat_interface(graph) -> None:
    """
    Render the chat interface in the right column.

    Supports multi-turn conversation with the LangGraph agent.
    Displays source citations with every response.
    Shows a clear "no relevant context" indicator when the
    hallucination guard fires.

    Parameters
    ----------
    graph : CompiledStateGraph
        The compiled LangGraph agent from get_compiled_graph().
    """
    st.subheader("💬 Interview Prep Chat")

    # Filters
    col_topic, col_diff = st.columns(2)
    with col_topic:
        # TODO: st.selectbox for topic filter
        pass
    with col_diff:
        # TODO: st.selectbox for difficulty filter
        pass

    # Chat history display
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    # Chat input
    # TODO: implement
    # 1. query = st.chat_input("Ask about a deep learning topic...")
    #
    # 2. On submit:
    #    a. Append user message to chat_history
    #    b. Display user message immediately (st.rerun or direct render)
    #    c. Build LangGraph input:
    #       {"messages": [HumanMessage(content=query)]}
    #    d. config = {"configurable": {"thread_id": st.session_state.thread_id}}
    #    e. result = graph.invoke(input, config=config)
    #    f. response = result["final_response"]
    #    g. Append assistant message with answer, sources, no_context_found flag
    #
    # STRETCH GOAL — streaming:
    # Replace graph.invoke with graph.stream() and use st.write_stream()
    # to display tokens as they arrive. Significant "wow factor" in Hour 3.


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_panel(graph)


if __name__ == "__main__":
    main()
