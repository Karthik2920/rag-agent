"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages

from rag_agent.agent.prompts import SYSTEM_PROMPT
from rag_agent.agent.state import AgentResponse, AgentState
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.

    Natural language questions are often poorly suited for vector
    similarity search. This node rephrases the query into a form
    that produces better embedding matches against the corpus.

    Example
    -------
    Input:  "cnn?"
    Output: "What is a Convolutional Neural Network and how does it work?"
    """
    settings = get_settings()
    llm = LLMFactory(settings).create()

    if isinstance(state, dict):
        messages = state.get("messages", [])
        original_query = state.get("original_query", "")
    else:
        messages = state.messages
        original_query = getattr(state, "original_query", "")

    last_message = messages[-1] if messages else None
    if isinstance(last_message, HumanMessage):
        original_query = last_message.content

    try:
        prompt = f"""Rewrite this query for semantic search in a vector database.
Make it concise and keyword-focused.

Query: {original_query}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()
        return {"original_query": original_query, "rewritten_query": rewritten}
    except Exception:
        return {"original_query": original_query, "rewritten_query": original_query}


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.

    Sets no_context_found=True if no chunks are returned, which triggers
    the hallucination guard in the conditional edge.
    """
    manager = VectorStoreManager()

    if isinstance(state, dict):
        rewritten_query = state.get("rewritten_query", "")
        topic_filter = state.get("topic_filter", None)
        difficulty_filter = state.get("difficulty_filter", None)
    else:
        rewritten_query = state.rewritten_query
        topic_filter = state.topic_filter
        difficulty_filter = state.difficulty_filter

    chunks = manager.query(
        query_text=rewritten_query,
        topic_filter=topic_filter,
        difficulty_filter=difficulty_filter,
    )

    if not chunks:
        return {"retrieved_chunks": [], "no_context_found": True}
    return {"retrieved_chunks": chunks, "no_context_found": False}


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.

    Implements the hallucination guard: if no_context_found is True,
    returns a clear fallback message rather than allowing the LLM to
    answer from parametric memory.
    """
    settings = get_settings()
    llm = LLMFactory(settings).create()

    if isinstance(state, dict):
        no_context = state.get("no_context_found", False)
        retrieved_chunks = state.get("retrieved_chunks", [])
        original_query = state.get("original_query", "")
        rewritten_query = state.get("rewritten_query", "")
    else:
        no_context = state.no_context_found
        retrieved_chunks = state.retrieved_chunks
        original_query = state.original_query
        rewritten_query = state.rewritten_query

    # ---- Hallucination Guard ------------------------------------------------
    if no_context:
        no_context_message = (
            "I was unable to find relevant information in the corpus for your query. "
            "This may mean the topic is not yet covered in the study material, or "
            "your query may need to be rephrased. Please try a more specific "
            "deep learning topic such as 'LSTM forget gate' or 'CNN pooling layers'."
        )
        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=rewritten_query,
        )
        return {
            "final_response": response,
            "messages": [AIMessage(content=no_context_message)],
        }

    # ---- Build Context from Retrieved Chunks --------------------------------
    context = ""
    sources = []
    scores = []
    for chunk in retrieved_chunks:
        citation = chunk.to_citation()
        context += f"{citation}\n{chunk.chunk_text}\n\n"
        sources.append(citation)
        scores.append(chunk.score)

    avg_confidence = sum(scores) / len(scores) if scores else 0.0

    # Build conversation history trimmed to max_context_tokens
    if isinstance(state, dict):
        history = state.get("messages", [])
    else:
        history = list(state.messages)

    try:
        trimmed_history = trim_messages(
            history,
            max_tokens=settings.max_context_tokens,
            strategy="last",
            token_counter=llm,
            include_system=True,
            allow_partial=False,
        )
    except Exception:
        trimmed_history = history[-6:] if len(history) > 6 else history

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"""Use the following context to answer the question. Answer directly and confidently based on what the context says. Do not add disclaimers or say you cannot answer if the context contains relevant information.

Context:
{context}

Question:
{original_query}

Rules:
- Answer using ONLY information found in the context above
- Do NOT use outside knowledge
- Be concise and direct (3–5 sentences)
- Only say you cannot answer if the context truly has NO relevant information at all"""
        ),
    ] + [m for m in trimmed_history if not isinstance(m, SystemMessage)]

    response = llm.invoke(messages)
    answer = response.content

    agent_response = AgentResponse(
        answer=answer,
        sources=sources,
        confidence=avg_confidence,
        no_context_found=False,
        rewritten_query=rewritten_query,
    )
    return {
        "final_response": agent_response,
        "messages": [AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Conditional edge: route to generation or end based on retrieval result.

    Returns "generate" if relevant chunks were found, "end" otherwise.
    This is the hallucination guard decision point in the graph.
    """
    if isinstance(state, dict):
        no_context = state.get("no_context_found", False)
    else:
        no_context = state.no_context_found

    return "end" if no_context else "generate"
