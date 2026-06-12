# from fastapi import FastAPI
# from pydantic import BaseModel
# from rag import initialize

# app = FastAPI(title="Wildfire Research RAG API")

# print("Initializing RAG pipeline...")
# rag_chain, retriever = initialize()
# print("RAG pipeline ready ✅")

# class QueryRequest(BaseModel):
#     question: str

# class QueryResponse(BaseModel):
#     answer: str

# @app.get("/health")
# def health():
#     return {"status": "ok", "message": "RAG API is running"}

# @app.post("/query", response_model=QueryResponse)
# def query(request: QueryRequest):
#     answer = rag_chain.invoke(request.question)
#     return QueryResponse(answer=answer)

# from rag import initialize
# rag_chain, retriever = initialize()

# # Check unique sources
# results = retriever.vectorstore.get()
# sources = set(m['source'] for m in results['metadatas'])
# for s in sources:
#     print(s)

from fastapi import FastAPI
from pydantic import BaseModel
from rag import initialize

app = FastAPI(title="Wildfire Research RAG API")

print("Initializing RAG pipeline...")
ask_question, retriever = initialize()
print("RAG pipeline ready ✅")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/health")
def health():
    return {"status": "ok", "message": "RAG API is running"}

# @app.post("/query", response_model=QueryResponse)
# def query(request: QueryRequest):
#     answer = ask_question(request.question)
#     return QueryResponse(answer=answer)

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    question = request.question.strip()
    if not question:
        return QueryResponse(answer="Please ask a question.")
    try:
        answer = ask_question(question)
        return QueryResponse(answer=answer)
    except Exception:
        return QueryResponse(answer="Sorry bro, something went wrong. Please try again in a moment.")