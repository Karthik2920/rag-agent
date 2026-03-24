# System Architecture
## Team: Group-5
## Date: 03/23/26
## Members and Roles:

- Corpus Architect: Karthik Saraf  
- Pipeline Engineer: Manoj Anandhan  
- UX Lead: Fidel Gonzales  
- Prompt Engineer: Sowmika Yeadhara  
- QA Lead: Karthik Saraf  

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
- **File formats used:** `.md`

- **Landmark papers ingested:**
  - CNN basics
  - RNN basics
  - ANN basics

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
| Normal query | Correct answer | Works | Pass |
| Off-topic query | Fallback | Works | Pass |
| Duplicate ingestion | Skipped | Works | Pass |
| Empty query | No crash | Works | Pass |
| Cross-topic query | Partial | Works | Pass |

---

## Known Limitations

- Limited dataset size  
- No advanced filtering  
- No PDF ingestion  

---

## What We Would Do With More Time

- Add hybrid search  
- Add re-ranking  
- Support PDFs  

---

## Hour 3 Interview Questions

**Question 1:** Why use RAG?  
**Answer:** It grounds responses in real data and prevents hallucination.

**Question 2:** Why ChromaDB?  
**Answer:** Lightweight and easy for local vector storage.

**Question 3:** How do you prevent hallucination?  
**Answer:** Strict prompts + fallback mechanism.

---

## Team Retrospective

**What clicked:**
- RAG pipeline design

**What confused us:**
- LangGraph integration

**Study next:**
- Advanced retrieval techniques