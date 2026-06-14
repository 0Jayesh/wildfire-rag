import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Paths
DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")

def classify_question(question: str, llm) -> str:
    """Classify question into: greeting, own_work, or general"""
    classification_prompt = f"""Classify this user message into exactly ONE category. Respond with only one word.

Categories:
- greeting: casual greetings, thanks, or asking what the assistant can do (e.g. "hi", "thanks", "who are you")
- own_work: asking about who the author is, what the title is, who supervised the research, when was it submitted, Jayesh Kumeriya's work — his research, project, dissertation, thesis, study, model, system, results, architecture, contributions, findings, achievements, scope, limitations, what it does or does not do, and whether it covers classification, detection, or segmentation. Includes generic phrases like "this research", "the project", "this work", "this study", "the paper", "the system", "does it do X" when context implies it's about the assistant's own subject matter. When in doubt, prefer own_work over general.
- general: asking about wildfire detection concepts broadly, other researchers' papers, comparisons, or techniques not specific to Jayesh's work

Message: "{question}"

Category (one word only):"""

    response = llm.invoke(classification_prompt)
    # category = response.content.strip().lower()
    match = re.search(r'\b(greeting|own_work|general)\b', response.content.lower())

    # Safety fallback - default to own_work if classification is unclear
    # if category not in {"greeting", "own_work", "general"}:
    #     return "own_work"
    # return category
    return match.group(1) if match else "own_work" 

def load_documents():
    """Load all PDFs from data folder using PyMuPDF"""
    all_pages = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDFs")
    
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        all_pages.extend(pages)
        print(f"Loaded {pdf_file.name} — {len(pages)} pages")
    
    print(f"Total pages: {len(all_pages)}")
    return all_pages

def create_chunks(pages):
    """Split pages into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(pages)
    print(f"Total chunks: {len(chunks)}")
    return chunks

def get_vectorstore(chunks=None):
    """Get or create ChromaDB vectorstore"""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # If ChromaDB already exists, load it
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print("Loading existing ChromaDB...")
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings
        )
        print(f"Loaded {vectorstore._collection.count()} vectors")
        return vectorstore
    
    # Otherwise create new one
    print("Creating new ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    print(f"Created {vectorstore._collection.count()} vectors")
    return vectorstore

def get_rag_chain(vectorstore):
    """Build the RAG components"""

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    prompt_template = """You are a research assistant presenting Jayesh Kumeriya's wildfire detection research (VNIT Nagpur, MTech Applied AI, FICTA 2026 publication).

Principles:
- Read EVERY chunk in the context carefully before answering - the most relevant information may appear in any chunk, not just the first one. Scan the entire context for specific numbers, lists, or named items (e.g. "three limitations", "four classes") before concluding the information isn't present.
- Ground every answer strictly in the provided context. Never add outside knowledge.
- If the context contains numbers, results, or specific findings, lead with those. These ARE the "achievements," "findings," "results," or "contributions" — whatever term the user uses.
- If a question is broad (e.g. "the architecture", "the model", "the approach") and the context contains multiple architectures, prioritize the Enhanced MHCNNFD (the main contribution) unless the context clearly indicates a different specific paper or technique was asked about.
- If asked whether the research does/includes something (e.g. "does it do segmentation", "is X covered"), check both what IS done AND the explicitly stated Scope/Limitations sections (which list exclusions). If the topic is listed as an exclusion, say so clearly.
- When citing, mention the document and page naturally (e.g. "as shown on page 35 of the dissertation"). Do not say whether something "is" or "is not" Jayesh's work — just cite the source.
- If the retrieved context doesn't contain the answer, say so plainly. Do not guess or substitute information from a different topic.
- Be concise and factual. Avoid hedging phrases like "it can be inferred" or "likely."

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    JAYESH_DOCS = [
        str(DATA_DIR / "FICTA2026_Kumeriya_WildfireDetection.pdf"),
        str(DATA_DIR / "REAL TIME WILDFIRE DETECTION SYSTEM USING - REPORT.pdf"),
    ]
    
    retriever_own = vectorstore.as_retriever(
        search_kwargs={"k": 10, "filter": {"source": {"$in": JAYESH_DOCS}}}
    )
    
    retriever_all = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    def format_docs(docs):
        return "\n\n".join([
            f"[Page {d.metadata.get('page', 'N/A')} - {Path(d.metadata.get('source', 'unknown')).name}]: {d.page_content}"
            for d in docs
        ])
    
    GREETING_RESPONSES = {
        "default": "Hi! I'm a research assistant for Jayesh Kumeriya's wildfire detection research (VNIT Nagpur, MTech Applied AI). Ask me about the architecture, datasets, results, or methodology."
    }

    METADATA = """
    Title: Real Time Wildfire Detection System Using Satellite Images and Deep Learning
    Author: Jayesh Kumeriya (MT24AAI032)
    Supervisor: Dr. Ankit A. Bhurane
    Institute: VNIT Nagpur, ECE Department
    Degree: MTech Artificial Intelligence
    Submission: May 2026
    Publication: FICTA 2026
    """
    
    def ask_question(question: str, chat_history: list = None) -> str:
        if chat_history is None:
            chat_history = []
        
        category = classify_question(question, llm)
        
        if category == "greeting":
            return GREETING_RESPONSES["default"]
        
         # If there's history, create a combined query for retrieval
        retrieval_query = question
        if chat_history:
            last_turn = chat_history[-1]
            retrieval_query = f"{last_turn['question']} {question}"
        
        retriever = retriever_own if category == "own_work" else retriever_all
        docs = retriever.invoke(retrieval_query)

        if category == "own_work":
            context = METADATA + "\n\n" + format_docs(docs)
        else:
            context = format_docs(docs)
        
        # Build history string
        history_text = ""
        if chat_history:
            history_text = "Previous conversation:\n"
            for turn in chat_history[-3:]:  # last 3 exchanges only
                history_text += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"
        
        final_prompt = prompt.format(
            context=history_text + "\n" + context, 
            question=question
        )
        response = llm.invoke(final_prompt)
        return response.content
    return ask_question, retriever_own

    # def ask_question(question: str) -> str:
    #     category = classify_question(question, llm)
        
    #     if category == "greeting":
    #         return GREETING_RESPONSES["default"]
        
    #     retriever = retriever_own if category == "own_work" else retriever_all
    #     docs = retriever.invoke(question)

    #     if category == "own_work":
    #         context = METADATA + "\n\n" + format_docs(docs)
    #     else:
    #         context = format_docs(docs)
        
    #     final_prompt = prompt.format(context=context, question=question)
    #     response = llm.invoke(final_prompt)
    #     return response.content
    # return ask_question, retriever_own

def initialize():
    """Full initialization — call this once on startup"""
    pages = load_documents()
    chunks = create_chunks(pages)
    vectorstore = get_vectorstore(chunks)
    ask_question, retriever = get_rag_chain(vectorstore)
    return ask_question, retriever