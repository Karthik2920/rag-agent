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
            import os

            file_paths = []
            # Save files temporarily
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                   tmp.write(file.getvalue())
                # from pathlib import Path

                file_paths.append((Path(tmp.name), file.name))
            # Chunk files
            chunks = chunker.chunk_files(file_paths)
            # Store in vector DB
            result = store.ingest(chunks)
            # Show results
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
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present. Used during Hour 3 to demonstrate corpus completeness.

    Parameters
    ----------
    store : VectorStoreManager
    """
    # TODO: implement
    # stats = store.get_collection_stats()
    # st.sidebar.metric("Total Chunks", stats["total_chunks"])
    # st.sidebar.write("Topics:", ", ".join(stats["topics"]))
    # if stats["bonus_topics_present"]:
    #     st.sidebar.success("✅ Bonus topics present")
    # else:
    #     st.sidebar.warning("⚠️ No bonus topics yet")
    pass


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


# def render_document_viewer(store: VectorStoreManager) -> None:
def render_document_viewer(store: VectorStoreManager) -> None:
    st.subheader("📄 Document Viewer")

    # Get all documents
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

    # Show all chunks
    for text, meta in zip(documents, metadatas):
        with st.expander(f"{meta.get('source', 'Unknown')} | {meta.get('topic')}"):
            st.write(text)
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    # st.subheader("📄 Document Viewer")

    # TODO: implement
    # 1. If no documents ingested: show placeholder message
    #
    # 2. st.selectbox("Select document", options=[doc["source"] for doc in docs])
    #    Store selection in st.session_state["selected_document"]
    #
    # 3. On selection change: store.get_document_chunks(selected_source)
    #
    # 4. Render chunks in a scrollable container (st.container with fixed height)
    #    For each chunk:
    #    - Show metadata badge: topic | difficulty | type
    #    - Show chunk text
    #    - Show similarity score if this chunk was used in last response
    #
    # 5. Display chunk count and coverage summary below viewer

    # st.info("Ingest documents using the sidebar to view content here.")


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# graph = get_compiled_graph()
def render_chat_panel(store):
    import streamlit as st

    st.subheader("💬 Interview Prep Chat")

    query = st.chat_input("Ask a question")

    if query:
        if not query:
            return
        results = store.query(query)

        if not results or len(results) == 0:
            st.error("⚠️ No relevant context found. Please ask a question related to the uploaded documents.")
            return
        valid_chunks = [r for r in results if len(r.chunk_text.strip()) > 30]
        top_chunk = results[0].chunk_text.lower()
        # if len(top_chunk) < 50:
        #     st.error("⚠️ Retrieved context is too weak to answer confidently.")
        #     return
        from rag_agent.config import LLMFactory
        llm = LLMFactory().create()
        # Show retrieved chunks
        with st.expander("📚 Retrieved Context"):
            for r in results:
                st.write(f"{r.metadata.topic}: {r.chunk_text[:200]}...")

        # Simple answer
        unique_chunks = list(dict.fromkeys([r.chunk_text for r in results]))
        context = "\n\n".join(unique_chunks[:2])

        
        

       
#         prompt = [
#              {"role": "system", "content": SYSTEM_PROMPT},
#              {
#                  "role": "user",
#                 "content": f"""
#         You are a deep learning interview assistant.

# STRICT RULES:
# - Answer ONLY using the provided context
# - If the answer is NOT clearly in the context, respond EXACTLY with:
#   "No relevant context found"
# - Do NOT use outside knowledge
# - Do NOT guess or assume
# - Keep the answer concise (3–5 sentences)
# - Include citation in format: [SOURCE: topic | filename]
#         Context:
#         {context}

#         Question:
#         {query}

#         Answer clearly and concisely.
#         """
#             }
#         ]
        user_prompt = ANSWER_PROMPT.format(
            context=context,
            question=query
        )

        # prompt = [
        #     {"role": "system", "content": SYSTEM_PROMPT},
        #     {"role": "user", "content": user_prompt}
        # ]
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ANSWER_PROMPT.format(
                    context=context,
                    question=query
                )       
            }
    ]
        # response = llm.invoke(prompt)
        # prompt = f"""
      # You are a deep learning assistant.

        # Answer the question using the context below.

        # Rules:
        # - Do NOT use any external knowledge
        # - Do NOT make assumptions  
        # - Do NOT include any source citations in your answer 
        # - Give ONLY one final answer
        # - Do NOT repeat information
        # - Keep it concise (3–5 sentences)

        # Context:
        # {context}

        # Question:
        # {query}

        # Answer:
        # """
        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)
        # sources = list(dict.fromkeys([
        #     f"{r.metadata.topic} | {r.metadata.source}" for r in results
        # ]))

        # source_text = "\n".join([f"[SOURCE: {s}]" for s in sources])
        answer = response.content
        answer_lower = answer.lower()
        st.chat_message("assistant").write(answer)
        # st.write("DEBUG RESULTS:", results)
        # final_answer = f"{response.content}\n\n{source_text}"
        # st.chat_message("assistant").write(final_answer)
        # st.write("### Answer")
        # st.write(response.content) 
        if not ("does not contain enough information" in answer_lower):
            st.markdown("### 📚 Sources")

    #     sources = list(dict.fromkeys([
    #         getattr(r.metadata, "source", "Unknown") for r in results
    # ]))

    #     for src in sources:
    #         st.markdown(f"- 📄 **{src}**")
        
        sources = list(dict.fromkeys([
            f"{r.metadata.source} ({r.metadata.topic})"
            for r in results[:2]
        ]))

        for src in sources:
            st.markdown(f"- 📄 **{src}**")
        # for r in results[:2]:
        #     st.markdown(f"- 📄 **{r.metadata.source}** ({r.metadata.topic})")
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
        render_chat_panel(store)


if __name__ == "__main__":
    main()
