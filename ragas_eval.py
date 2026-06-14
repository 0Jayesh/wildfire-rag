import os
from dotenv import load_dotenv
from rag import initialize
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

print("Initializing RAG pipeline...")
ask_question, retriever = initialize()

# Test questions with ground truth answers
test_data = [
    {
        "question": "What accuracy did the Enhanced MHCNNFD achieve on UAVs-FFDB?",
        "ground_truth": "The Enhanced MHCNNFD achieved 100.00% test accuracy on UAVs-FFDB."
    },
    {
        "question": "What three architectural limitations led to the enhanced version?",
        "ground_truth": "Insufficient filter capacity, absent Batch Normalisation, and weak regularisation."
    },
    {
        "question": "What is the fog degradation formula?",
        "ground_truth": "I_fog = 0.3 x I_original + 0.7 x (180 x J)"
    },
    {
        "question": "Who supervised this dissertation?",
        "ground_truth": "Dr. Ankit A. Bhurane supervised this dissertation."
    },
]

print("Running questions...")
results = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

for item in test_data:
    q = item["question"]
    answer = ask_question(q)
    docs = retriever.invoke(q)
    contexts = [d.page_content for d in docs]
    
    results["question"].append(q)
    results["answer"].append(answer)
    results["contexts"].append(contexts)
    results["ground_truth"].append(item["ground_truth"])
    print(f"Q: {q}\nA: {answer[:100]}...\n")

dataset = Dataset.from_dict(results)

print("Evaluating with RAGAS...")

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

score = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

print("\n=== RAGAS Scores ===")
print(score)

df = score.to_pandas()
df.to_csv("ragas_results.csv", index=False)
print("\nSaved to ragas_results.csv")