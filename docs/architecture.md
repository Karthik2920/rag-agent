# System Architecture
## Team: Group-5
## Date: 03/28/26
## Members and Roles:

- Corpus Architect: Karthik Saraf
- Pipeline Engineer: Manoj Anandhan
- UX Lead: Fidel Gonzales
- Prompt Engineer: Sowmika Yeadhara
- QA Lead: Akshaya Paila

---

## Architecture Diagram

```
User Query
   │
   ▼
┌─────────────────────┐
│  Query Rewrite Node │  Clarifies vague or short queries using LLM
└─────────────────────┘
   │
   ▼
┌─────────────────────┐
│   Retrieval Node    │  Embeds rewritten query → queries ChromaDB (top-5)
└─────────────────────┘
   │
   ▼
┌──────────────────────────────────┐
│  Relevant Context Found?         │
│  Score threshold + content check │
└──────────────────────────────────┘
   │ YES                    │ NO
   ▼                        ▼
┌──────────────┐     ┌─────────────────────┐
│ Generation   │     │  Hallucination Guard │
│    Node      │     │  Returns fallback    │
└──────────────┘     └─────────────────────┘
   │
   ▼
Final Answer + Source Citations

─────────────────────────────────────

Corpus Ingestion Flow:

Upload (PDF / MD)
   │
   ▼
DocumentChunker
   │  ├─ PyPDFLoader (PDF)
   │  └─ Paragraph split (MD)
   ▼
RecursiveCharacterTextSplitter (512 chars, 50 overlap)
   │
   ▼
SHA-256 Duplicate Check
   │
   ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
   │
   ▼
ChromaDB Vector Store (cosine similarity)
```

---

## Component Descriptions

### Corpus Layer

- **Source files location:** `data/corpus/`
- **File formats used:** `.md`, `.pdf`

- **Documents ingested (12 total):**

| File | Topic | Type |
|---|---|---|
| alexnet.pdf | CNN | Landmark paper |
| ann.pdf | ANN | Landmark paper |
| lstm.pdf | LSTM | Landmark paper |
| ann_intermediate.md | ANN | Concept notes |
| cnn_intermediate.md | CNN | Concept notes |
| lstm_intermediate.md | LSTM | Concept notes |
| rnn_intermediate.md | RNN | Concept notes |
| seq2seq_intermediate.md | Seq2Seq | Concept notes |
| autoencoder_intermediate.md | Autoencoder | Concept notes |
| som_intermediate.md | SOM | Concept notes (bonus) |
| boltzmann_intermediate.md | Boltzmann Machine | Concept notes (bonus) |
| gan_intermediate.md | GAN | Concept notes (bonus) |

- **Chunking strategy:**
  512 characters with 50-character overlap. Overlap prevents concepts that span chunk boundaries from being lost entirely — a common interview talking point.

- **Metadata schema:**

| Field | Type | Purpose |
|---|---|---|
| topic | string | Identify subject (CNN, RNN, etc.) |
| difficulty | string | Beginner/Intermediate |
| type | string | Concept explanation |
| source | string | File name |
| related_topics | list | Future extension |
| is_bonus | bool | Advanced topics (SOM, GAN, etc.) |

- **Duplicate detection approach:**
  Content-based SHA-256 hashing on `source + chunk_text`. This ensures the same file re-uploaded produces the same IDs, so duplicates are skipped regardless of filename changes.

- **Corpus coverage:**
- [x] ANN
- [x] CNN
- [x] RNN
- [x] LSTM
- [x] Seq2Seq
- [x] Autoencoder
- [x] SOM
- [x] Boltzmann Machine
- [x] GAN

---

### Vector Store Layer

- **Database:** ChromaDB (PersistentClient)
- **Local persistence path:** `./chroma_db`

- **Embedding model:**
`all-MiniLM-L6-v2` (HuggingFace sentence-transformers)

- **Why this embedding model:**
Fast, lightweight, and optimised for semantic similarity. Runs locally with no API cost.

- **Similarity metric:**
Cosine similarity (`hnsw:space = cosine`)

- **Retrieval k:**
Top 5 chunks per query

- **Similarity threshold:**
Implicit — fallback fires when retrieved chunks score low and context is insufficient for answering

- **Metadata filtering:**
Topic and difficulty filters available via `where=` parameter in ChromaDB; exposed as optional UI controls

---

### Agent Layer

- **Framework:** LangGraph (StateGraph)

- **State schema:** `AgentState` (TypedDict)

| Field | Type | Description |
|---|---|---|
| messages | list | LangChain message history |
| rewritten_query | str | Clarified version of user query |
| retrieved_chunks | list | Top-k chunks from ChromaDB |
| no_context_found | bool | Hallucination guard flag |
| final_response | AgentResponse | Structured answer with sources |

- **Graph nodes:**

| Node | Responsibility |
|---|---|
| query_rewrite_node | Rewrites vague queries using LLM for better retrieval |
| retrieval_node | Embeds rewritten query and fetches top-5 chunks from ChromaDB |
| generation_node | Generates grounded answer using retrieved context |

- **Conditional edges:**
After retrieval, if `no_context_found = True` → skip generation, return hallucination guard message directly.

- **Hallucination guard:**
"The provided context does not contain enough information to answer this question."

- **Query rewriting example:**
  - Raw: `"cnn?"`
  - Rewritten: `"What is a Convolutional Neural Network and how does it work?"`

- **Conversation memory:**
Handled via `MemorySaver` (in-memory, keyed by `thread_id` per session)

- **LLM provider:**
Groq / OpenAI-compatible API (llama-3 or compatible model)

- **Why this provider:**
Fast inference speed and OpenAI-compatible API — minimal code changes if switching providers.

---

### Prompt Layer

- **System prompt design:**
Strict context-only assistant — explicitly instructed not to use prior knowledge. Any question that cannot be answered from retrieved chunks triggers the hallucination guard.

- **Query rewrite prompt:**
Uses the raw user query as input and asks the LLM to produce a clearer, more complete version suitable for semantic search.

- **Generation prompt:**
Combines rewritten query + retrieved chunk texts. Instructs the LLM to cite sources and admit uncertainty when context is insufficient.

- **Structured output:**
`AgentResponse` is a Pydantic model — ensures `answer`, `sources`, `confidence`, and `no_context_found` fields are always present and typed correctly.

- **Failure modes identified and mitigated:**

| Failure | Mitigation |
|---|---|
| Hallucination | Strict context-only prompt + hallucination guard node |
| Weak context | `no_context_found` flag → fallback message |
| Repetition | Controlled via prompt instructions |
| Off-topic queries | Hallucination guard fires; no sources shown |

---

### Interface Layer

- **Framework:** Streamlit
- **Deployment platform:** Streamlit Community Cloud — https://rag-agent-ccmufe7hlptmlcrvlsjin4.streamlit.app/

- **Sidebar — Corpus Ingestion:**
  - Multi-file upload (PDF and MD)
  - Progress bar during chunking and embedding
  - Success/error result display (chunks added, duplicates skipped)

- **Sidebar — Corpus Stats Badge:**
  - Shows total topics and chunk count at a glance
  - Lists all active topics as inline code badges

- **Sidebar — Corpus Library Panel:**
  - Color-coded cards per document (unique color per topic)
  - Shows source name, topic badge, chunk count
  - 📄 View button — auto-expands document in viewer panel
  - 🗑 Remove button — deletes all chunks for that document from ChromaDB

- **Centre — Document Viewer:**
  - Groups chunks by source document
  - Expandable per-document sections showing chunk previews
  - Clicking View in sidebar auto-expands the selected document

- **Right — Chat Interface:**
  - Example starter questions on empty state
  - Multi-turn conversation with timestamp per message
  - Source citations shown per answer (expandable)
  - Hallucination guard warning when no relevant context found
  - Clear Chat button to reset session

- **Session state keys:**

| Key | Stores |
|---|---|
| chat_history | Full conversation with timestamps and sources |
| selected_document | Document currently highlighted in viewer |
| thread_id | LangGraph conversation thread ID |
| topic_filter | Active topic filter for retrieval |
| difficulty_filter | Active difficulty filter for retrieval |

---

## Design Decisions

1. **Chunk size: 512 characters**
   **Rationale:** Balance between context richness and retrieval precision.
   **Interview answer:** "We chose 512 to retain one complete idea per chunk while keeping retrieval focused. A larger chunk like 1024 would retrieve the right document but deliver noisy context to the LLM."

2. **Chunk overlap: 50 characters**
   **Rationale:** Prevent concepts that span chunk boundaries from being split.
   **Interview answer:** "Without overlap, a sentence that straddles two chunks would be incomplete in both. 50-character overlap ensures no idea is lost at boundaries."

3. **Top-k = 5**
   **Rationale:** Enough context for cross-topic queries without flooding the prompt.
   **Interview answer:** "We retrieve 5 chunks to handle questions that span multiple topics (e.g., LSTM vs RNN), while keeping the context window focused."

4. **Strict context-only prompt**
   **Rationale:** Prevent hallucination — the system must not generate answers from prior training knowledge.
   **Interview answer:** "We enforce context-only answers so the system fails safely. A wrong answer from training data is worse than a transparent 'I don't know'."

5. **Content-based deduplication (SHA-256)**
   **Rationale:** Filename-based deduplication fails when files are renamed or re-uploaded.
   **Interview answer:** "We hash the source name and chunk text together. Two uploads of the same file always produce the same IDs, so ChromaDB's upsert naturally skips duplicates."

---

## QA Test Results

| Test | Expected | Actual | Pass / Fail |
|---|---|---|---|
| Normal query ("What is backpropagation?") | Correct answer with sources | Answer generated with source citations | Pass |
| Off-topic query ("What is the capital of France?") | Hallucination guard fires | "Does not contain enough information", no sources shown | Pass |
| Duplicate ingestion | 0 chunks added, all skipped | 0 chunks added, 453 duplicates skipped | Pass |
| Empty query | No crash | Streamlit blocks empty submission natively | Pass |
| Cross-topic query ("How do LSTMs improve on RNNs?") | Chunks from multiple topics | Retrieved from lstm.pdf and rnn_intermediate.md | Pass |
| Document delete | Chunks removed from ChromaDB | Document removed, corpus stats updated | Pass |

---

## Known Limitations

- No streaming responses — full answer appears after generation completes
- Metadata filtering (topic/difficulty) available in backend but not exposed as active UI controls
- PDF ingestion quality depends on PyPDFLoader text extraction; equations and figures are not captured
- In-memory conversation history resets on app restart (MemorySaver not persisted to disk)

---

## What We Would Do With More Time

- Add hybrid search (BM25 + semantic) for better keyword matching
- Add re-ranking (e.g., CrossEncoder) for improved retrieval precision
- ~~Add bonus topics: SOM, Boltzmann Machine, GAN~~ ✅ Completed
- Implement streaming token-by-token responses in the UI
- Persist conversation history across sessions using SQLite or file-based checkpointer
- Add metadata filtering UI controls (topic and difficulty dropdowns in chat panel)

---

## Hour 3 Interview Questions

**Question 1 (Single topic — LSTM):** Walk me through the three gates in an LSTM and what each one controls.

**Model Answer:** An LSTM has three gates. The forget gate decides what information to discard from the cell state — it outputs values between 0 and 1, where 0 means completely forget and 1 means completely keep. The input gate decides what new information to write into the cell state. The output gate controls what part of the cell state gets passed to the next hidden state. Together, these gates allow the LSTM to selectively remember or forget information at each time step, which is how it solves the vanishing gradient problem that standard RNNs suffer from.

---

**Question 2 (Cross-topic — Seq2Seq + Autoencoder):** How does the encoder in a Seq2Seq model relate to the encoder in an autoencoder?

**Model Answer:** Both encoders compress input into a lower-dimensional representation. In a Seq2Seq model, the encoder reads an input sequence and compresses it into a context vector, which the decoder uses to generate the output sequence. In an autoencoder, the encoder compresses the input into a latent space bottleneck, and the decoder reconstructs the original input. The key difference is purpose — Seq2Seq encodes for translation or generation of a different sequence, while an autoencoder encodes for reconstruction and feature learning. Both face the same bottleneck problem: too small a representation loses information.

---

**Question 3 (System design / tradeoff):** Why did your team choose chunk size 512 and what would break if you doubled it to 1024?

**Model Answer:** We chose 512 characters to balance context richness with retrieval precision. A chunk needs to be large enough to contain one complete idea, but small enough that when retrieved it is actually relevant to the query. If we doubled to 1024, each chunk would contain multiple ideas — retrieval would still find the chunk, but the LLM would receive noisy context with irrelevant content mixed in, degrading answer quality. It would also reduce the total number of chunks, meaning less granular retrieval. Smaller k values like our top-5 retrieval work well at 512 but would need to increase at 1024 to cover the same semantic ground.

---

**Question 4 (RAG pipeline):** What happens in your system when a user asks a question that is not covered by the corpus?

**Model Answer:** The query still goes through the full pipeline. The query rewrite node clarifies it, and the retrieval node fetches the top-5 most similar chunks. However, if those chunks do not contain relevant information, the generation node detects this and sets `no_context_found = True`. The system then returns the hallucination guard message — "The provided context does not contain enough information to answer this question" — and no sources are shown. This is intentional: a transparent failure is better than a confident wrong answer.

---

## Team Retrospective

**What clicked:**
- RAG pipeline design — retrieval + generation separation was clean
- ChromaDB integration was straightforward with good documentation
- Content-based deduplication worked reliably across multiple test ingestions
- LangGraph conditional edges made the hallucination guard easy to wire in

**What confused us:**
- LangGraph state management — passing data between nodes required careful TypedDict design
- PDF ingestion quality — PyPDFLoader extracts raw text which needed testing with real papers
- Streamlit rerun behavior — session_state timing required careful handling to avoid double reruns

**Study next:**
- Advanced retrieval techniques (hybrid search, re-ranking)
- LangGraph streaming and async graph execution
- Variational autoencoders and generative model internals
