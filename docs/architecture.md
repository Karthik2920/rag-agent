# System Architecture
## Team: Group-5
## Date: 03/23/26
## Members and Roles:

- Corpus Architect: Karthik Saraf  
- Pipeline Engineer: Manoj Anandhan  
- UX Lead: Fidel Gonzales  
- Prompt Engineer: Sowmika Yeadhara  
- QA Lead: Akshaya Paila  

---

## Architecture Diagram

User Query  
   ↓  
Query Rewrite Node  
   ↓  
Retrieval Node (ChromaDB)  
   ↓  
[Check: Relevant Context?]  
   ↓ YES → Generation Node → Final Answer  
   ↓ NO  → Hallucination Guard → "Not enough information"  

Corpus Flow:  
File → Chunking → Embedding → Vector Store (ChromaDB)

---

## Component Descriptions

### Corpus Layer

- **Source files location:** `data/corpus/`
- **File formats used:** `.md`, `.pdf`

- **Landmark papers ingested:**
- CNN basics (PDF)
- RNN basics (PDF)
- ANN basics (PDF)

- **Chunking strategy:**
  512 characters with small overlap for better semantic retrieval.

- **Metadata schema:**

| Field | Type | Purpose |
|---|---|---|
| topic | string | Identify subject (CNN, RNN, etc.) |
| difficulty | string | Beginner/Intermediate |
| type | string | Concept explanation |
| source | string | File name |
| related_topics | list | Future extension |
| is_bonus | bool | Advanced topics |

- **Duplicate detection approach:**
Content-based hashing ensures duplicate chunks are skipped reliably.

- **Corpus coverage:**
- [x] ANN
- [x] CNN
- [x] RNN
- [ ] LSTM
- [ ] Seq2Seq
- [ ] Autoencoder
- [ ] SOM
- [ ] Boltzmann Machine
- [ ] GAN

---

### Vector Store Layer

- **Database:** ChromaDB  
- **Local persistence path:** `./chroma_db`

- **Embedding model:**
`all-MiniLM-L6-v2`

- **Why this embedding model:**
Fast, lightweight, and suitable for semantic search.

- **Similarity metric:**
Cosine similarity

- **Retrieval k:**
Top 2 chunks

- **Similarity threshold:**
Implicit threshold based on chunk quality

- **Metadata filtering:**
Not implemented (basic retrieval only)

---

### Agent Layer

- **Framework:** LangGraph

- **Graph nodes:**

| Node | Responsibility |
|---|---|
| query_rewrite_node | Improves query clarity |
| retrieval_node | Fetches relevant chunks |
| generation_node | Generates answer |

- **Conditional edges:**
If no relevant context → return fallback message

- **Hallucination guard:**
"The provided context does not contain enough information to answer this question."

- **Query rewriting:**
- Raw: "cnn?"
- Rewritten: "What is a Convolutional Neural Network?"

- **Conversation memory:**
Handled via MemorySaver (in-memory session tracking)

- **LLM provider:**
Groq / OpenAI-compatible API

- **Why this provider:**
Fast inference and easy integration

---

### Prompt Layer

- **System prompt summary:**
Strict assistant that only answers using context

- **Question generation prompt:**
Uses context + query

- **Answer evaluation prompt:**
Ensures correctness and prevents hallucination

- **JSON reliability:**
Structured prompts used

- **Failure modes identified:**
- Hallucination → prevented via strict prompt
- Weak context → fallback response
- Repetition → controlled via prompt

---

### Interface Layer

- **Framework:** Streamlit  
- **Deployment platform:** Local  

- **Ingestion panel features:**
Upload and ingest documents with duplicate detection

- **Document viewer features:**
View ingested content and chunks

- **Chat panel features:**
Ask questions, see answers + sources, hallucination guard

- **Session state keys:**

| Key | Stores |
|---|---|
| chat_history | Conversation |
| ingested_documents | Files |
| selected_document | Active doc |
| thread_id | Session |

---

## Design Decisions

1. **Chunk size: 512**
   **Rationale:** Balance between context and precision  
   **Interview answer:** "We chose 512 to retain semantic meaning while avoiding noise."

2. **Top-k = 2**
   **Rationale:** Avoid irrelevant context  
   **Interview answer:** "Smaller k improves precision and reduces hallucination risk."

3. **Strict prompt**
   **Rationale:** Prevent hallucination  
   **Interview answer:** "We enforce context-only answers to ensure reliability."

---

## QA Test Results

| Test | Expected | Actual | Pass / Fail |
|---|---|---|---|
| Normal query ("What is backpropagation?") | Correct answer with sources | Answer generated with source citations | Pass |
| Off-topic query ("What is the capital of France?") | Hallucination guard fires | "Does not contain enough information", no sources shown | Pass |
| Duplicate ingestion | 0 chunks added, all skipped | 0 chunks added, 426 duplicates skipped | Pass |
| Empty query | No crash | Streamlit blocks empty submission natively | Pass |
| Cross-topic query ("How do LSTMs improve on RNNs?") | Chunks from multiple topics | Retrieved from lstm.pdf and rnn_intermediate.md | Pass |

---

## Known Limitations

- Limited dataset size  
- No advanced filtering  
- Basic PDF ingestion supported 

---

## What We Would Do With More Time

- Add hybrid search  
- Add re-ranking  
- Support PDFs  

---

## Hour 3 Interview Questions

**Question 1 (Single topic — LSTM):** Walk me through the three gates in an LSTM and what each one controls.

**Model Answer:** An LSTM has three gates. The forget gate decides what information to discard from the cell state — it outputs values between 0 and 1, where 0 means completely forget and 1 means completely keep. The input gate decides what new information to write into the cell state. The output gate controls what part of the cell state gets passed to the next hidden state. Together, these gates allow the LSTM to selectively remember or forget information at each time step, which is how it solves the vanishing gradient problem that standard RNNs suffer from.

---

**Question 2 (Cross-topic — Seq2Seq + Autoencoder):** How does the encoder in a Seq2Seq model relate to the encoder in an autoencoder?

**Model Answer:** Both encoders compress input into a lower-dimensional representation. In a Seq2Seq model, the encoder reads an input sequence and compresses it into a context vector, which the decoder uses to generate the output sequence. In an autoencoder, the encoder compresses the input into a latent space bottleneck, and the decoder reconstructs the original input. The key difference is purpose — Seq2Seq encodes for translation or generation of a different sequence, while an autoencoder encodes for reconstruction and feature learning. Both face the same bottleneck problem: too small a representation loses information.

---

**Question 3 (System design / tradeoff):** Why did your team choose chunk size 512 and what would break if you doubled it to 1024?

**Model Answer:** We chose 512 characters to balance context richness with retrieval precision. A chunk needs to be large enough to contain one complete idea, but small enough that when retrieved it is actually relevant to the query. If we doubled to 1024, each chunk would contain multiple ideas — retrieval would still find the chunk, but the LLM would receive noisy context with irrelevant content mixed in, degrading answer quality. It would also reduce the total number of chunks, meaning less granular retrieval. Smaller k values like our top-2 retrieval work well at 512 but would need to increase at 1024 to cover the same semantic ground.

---

## Team Retrospective

**What clicked:**
- RAG pipeline design

**What confused us:**
- LangGraph integration

**Study next:**
- Advanced retrieval techniques