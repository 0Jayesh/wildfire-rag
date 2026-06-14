# Wildfire Research Assistant

A domain-specific RAG (Retrieval-Augmented Generation) assistant built over published wildfire detection research.

[Live Demo →](https://huggingface.co/spaces/jayeshkumeriya/wildfire-rag)

## What it does

Ask questions about the research — architecture, datasets, results, methodology — and get precise, cited answers grounded in the actual papers (no hallucination). 

Example questions:
* "What three architectural limitations led to the Enhanced MHCNNFD?"
* "What datasets were used and how were they combined?"
* "What is the fog degradation formula?"
  
## Architecture

```
User question
    ↓
Question classifier (greeting / own_work / general)
    ↓
Two-tier retrieval:
  - own_work → Jayesh's 2 docs only (filtered ChromaDB)
  - general  → all 6 docs (includes literature review)
    ↓
Context + metadata → LLM (Groq llama-3.1-8b-instant)
    ↓
Cited answer
```

## Tech Stack

* LangChain — RAG orchestration
* ChromaDB — vector store (persistent)
* HuggingFace Embeddings — all-MiniLM-L6-v2 (384d)
* Groq — LLM inference (llama-3.1-8b-instant)
* FastAPI — backend API
* Streamlit — chat UI
* Docker — containerized deployment
- HuggingFace Spaces — hosting

## Knowledge Base

6 documents, 702 chunks:

* MTech Dissertation
* FICTA 2026 accepted paper
- 4 related literature review papers (MHCNNFD, DeepFire, FireDetn, SegNet)

## Run Locally

```bash
git clone https://github.com/0Jayesh/wildfire-rag
cd wildfire-rag
pip install -r requirements.txt
# Add GROQ_API_KEY to .env file
echo "GROQ_API_KEY=your_key_here" > .env
# Terminal 1
uvicorn main:app --reload
# Terminal 2
streamlit run app.py

```

Future Improvements

* Hybrid search (BM25 + semantic) for exact-term retrieval
* Cross-encoder re-ranking for better chunk prioritization
* Conversational memory for follow-up questions
- RAGAS-based automated evaluation

## Key Design Decisions

* PyPDFLoader chosen over PyMuPDF — empirically tested, gave more accurate retrieval for this document set
* Two-tier retrieval — prevents cross-document contamination (e.g. attributing other authors' work to Jayesh)
* Principles-based prompting — instructs the LLM to scan all retrieved chunks and ...ground answers strictly in context

---

[GitHub](https://github.com/0Jayesh/wildfire-rag)*
