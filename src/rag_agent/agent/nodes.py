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

from rag_agent.agent.prompts import (
    QUESTION_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import AgentResponse, AgentState, RetrievedChunk
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    settings = get_settings()
    llm = LLMFactory(settings).create()
    try:
        # Get latest user message
        # last_message = state.messages[-1]
        if isinstance(state, dict):
            messages = state.get("messages", [])
            original_query = state.get("original_query", "")
        else:
            messages = state.messages
            original_query = getattr(state, "original_query", "")

        # Get latest user message if available
        last_message = messages[-1] if messages else None
        if isinstance(last_message, HumanMessage):
            original_query = last_message.content
        # if isinstance(last_message, HumanMessage):
        #     original_query = last_message.content
        # else:
        #     original_query = state.original_query
        # Simple rewrite prompt
        prompt = f"""
Rewrite this query for semantic search in a vector database.
Make it concise and keyword-focused.

Query: {original_query}
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()
        return {
            "original_query": original_query,
            "rewritten_query": rewritten,
        }
    except Exception:
        return {
            "original_query": original_query,
            "rewritten_query": original_query,
        }
    
    """
    Rewrite the user's query to maximise retrieval effectiveness.

    Natural language questions are often poorly suited for vector
    similarity search. This node rephrases the query into a form
    that produces better embedding matches against the corpus.

    Example
    -------
    Input:  "I'm confused about how LSTMs remember things long-term"
    Output: "LSTM long-term memory cell state forget gate mechanism"

    Interview talking point: query rewriting is a production RAG pattern
    that significantly improves retrieval recall. It acknowledges that
    users do not phrase queries the way documents are written.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: messages (for context).

    Returns
    -------
    dict
        Updates: original_query, rewritten_query.
    """
    # TODO: implement
    # 1. Extract the latest HumanMessage from state.messages as original_query
    # 2. Build a short prompt instructing the LLM to rewrite for vector search
    #    Keep the rewriting prompt lightweight — this adds latency
    # 3. Call llm.invoke() with the rewrite prompt
    # 4. Return {"original_query": original_query, "rewritten_query": rewritten}
    #
    # Fallback: if rewriting fails (API error, timeout), return the original
    # query unchanged so the graph continues gracefully
    # raise NotImplementedError


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    manager = VectorStoreManager()
    # ✅ Handle both dict and AgentState
    if isinstance(state, dict):
        rewritten_query = state.get("rewritten_query", "")
        topic_filter = state.get("topic_filter", None)
        difficulty_filter = state.get("difficulty_filter", None)
    else:
        rewritten_query = state.rewritten_query
        topic_filter = state.topic_filter
        difficulty_filter = state.difficulty_filter

    chunks = manager.query(
        query_text=state.rewritten_query,
        topic_filter=state.topic_filter,
        difficulty_filter=state.difficulty_filter,
    )
    if not chunks:
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
        }
    return {
        "retrieved_chunks": chunks,
        "no_context_found": False,
    }

    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.

    Sets the no_context_found flag if no chunks meet the similarity
    threshold. This flag is checked by generation_node to trigger
    the hallucination guard.

    Interview talking point: separating retrieval into its own node
    makes it independently testable and replaceable — you could swap
    ChromaDB for Pinecone or Weaviate by changing only this node.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: rewritten_query, topic_filter, difficulty_filter.

    Returns
    -------
    dict
        Updates: retrieved_chunks, no_context_found.
    """
    # TODO: implement
    # 1. Instantiate VectorStoreManager (consider caching this)
    # 2. manager.query(
    #        query_text=state.rewritten_query,
    #        topic_filter=state.topic_filter,
    #        difficulty_filter=state.difficulty_filter
    #    )
    # 3. If result is empty: return {"retrieved_chunks": [], "no_context_found": True}
    # 4. Otherwise: return {"retrieved_chunks": chunks, "no_context_found": False}
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.

    Implements the hallucination guard: if no_context_found is True,
    returns a clear "no relevant context" message rather than allowing
    the LLM to answer from parametric memory.

    Implements token-aware conversation memory trimming: when the
    message history approaches max_context_tokens, the oldest
    non-system messages are removed.

    Interview talking point: the hallucination guard is the most
    commonly asked about production RAG pattern. Interviewers want
    to know how you prevent the model from confidently making up
    information when the retrieval step finds nothing relevant.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: retrieved_chunks, no_context_found, messages,
               original_query, topic_filter.

    Returns
    -------
    dict
        Updates: final_response, messages (with new AIMessage appended).
    """
    settings = get_settings()
    llm = LLMFactory(settings).create()
     # ✅ Handle dict vs AgentState
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

    # ---- Hallucination Guard -----------------------------------------------
    if state.no_context_found:
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
            rewritten_query=state.rewritten_query,
        )
        return {
            "final_response": response,
            "messages": [AIMessage(content=no_context_message)],
        }

    # ---- Build Context from Retrieved Chunks --------------------------------
    # TODO: implement
    # 1. Format retrieved chunks into a context string with citations
    #    Each chunk should appear as: "[SOURCE: topic | file]\n{chunk_text}\n"
    # 2. Calculate average confidence score from chunk scores
    # 3. Build the full prompt:
    #    - SystemMessage with SYSTEM_PROMPT
    #    - Context message with formatted chunks
    #    - Trimmed conversation history (trim to max_context_tokens)
    #    - HumanMessage with original_query
    # 4. llm.invoke(messages)
    # 5. Construct AgentResponse with answer, sources (list of citations), confidence
    # 6. Append AIMessage to messages
    # 7. Return {"final_response": response, "messages": [new_ai_message]}
    context = ""
    sources = []
    scores = []
    for chunk in retrieved_chunks:
        citation = chunk.to_citation()
        context += f"{citation}\n{chunk.chunk_text}\n\n"
        sources.append(citation)
        scores.append(chunk.score)
    avg_confidence = sum(scores) / len(scores) if scores else 0.0
    # ---- Build messages ----
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"""
Context:
{context}

Question:
{original_query}

Instructions:
- Answer ONLY using the provided context
- Do NOT use outside knowledge
- Do NOT infer anything not explicitly stated
- If answer is incomplete, say:
  "The provided context does not contain enough information to answer this question."
- Be concise (3–5 sentences)
"""
        ),
    ]
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
    if isinstance(state, dict):
        no_context = state.get("no_context_found", False)
    else:
        no_context = state.no_context_found
    if no_context:
        return "end"
    return "generate"

    """
    Conditional edge function: decide whether to retry retrieval or generate.

    Called by the graph after retrieval_node. If no context was found,
    the graph routes back to query_rewrite_node for one retry with a
    broader query before triggering the hallucination guard.

    Interview talking point: conditional edges in LangGraph enable
    agentic behaviour — the graph makes decisions about its own
    execution path rather than following a fixed sequence.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: no_context_found, retrieved_chunks.

    Returns
    -------
    str
        "generate" — proceed to generation_node.
        "end"      — skip generation, return no_context response directly.

    Notes
    -----
    Retry logic should be limited to one attempt to prevent infinite loops.
    Track retry count in AgentState if implementing retry behaviour.
    """
    # TODO: implement
    # Simple version: if no_context_found → "end", else → "generate"
    # Advanced version: track retry count, allow one retry with broader query
    # raise NotImplementedError
