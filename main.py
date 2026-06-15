from fastapi import FastAPI
from pydantic import BaseModel
from rag import initialize

app = FastAPI(title="Wildfire Research RAG API")

print("Initializing RAG pipeline...")
ask_question, retriever = initialize()
print("RAG pipeline ready")

class QueryRequest(BaseModel):
    question: str
    chat_history: list = []

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
        # answer = ask_question(question)
        answer = ask_question(question, request.chat_history)
        return QueryResponse(answer=answer)
    except Exception:
        return QueryResponse(answer="Sorry bro, something went wrong. Please try again in a moment.")