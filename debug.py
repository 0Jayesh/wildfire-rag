from rag import initialize  # or whatever your file is named
from pathlib import Path

ask_question, retriever = initialize()

questions = [
    "What three architectural limitations were identified that led to the development of the enhanced version?",
    "What specific grouping constraint was enforced during the partitioning of the Robust Augmented Dataset?",
    "What does FADE by Phan et al. refer to?"
]

for q in questions:
    docs = retriever.invoke(q)
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print(f"Chunks retrieved: {len(docs)}")
    for i, d in enumerate(docs):
        print(f"\n[Chunk {i+1}] Page {d.metadata.get('page','?')} - {Path(d.metadata.get('source','?')).name}")
        print(d.page_content[:300])
    print(f"{'='*60}")