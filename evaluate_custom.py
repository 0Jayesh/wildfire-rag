import os
import time
import json
import mlflow
import pandas as pd
from dotenv import load_dotenv
from rag import initialize
from groq import Groq

load_dotenv()

ask_question, retriever = initialize()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

def score_answer(question, answer, contexts, ground_truth):
    prompt = f"""You are evaluating a RAG system. Score the answer on two metrics.

Question: {question}
Retrieved Context (first 1000 chars): {contexts[0][:1000] if contexts else 'None'}
Answer: {answer}
Ground Truth: {ground_truth}

Score these metrics from 0.0 to 1.0:
1. faithfulness: Is the answer grounded only in the retrieved context? (1.0 = fully grounded, 0.0 = hallucinated)
2. answer_relevance: Does the answer actually address the question asked? (1.0 = fully answers it, 0.0 = irrelevant)

Respond with only valid JSON like this:
{{"faithfulness": 0.9, "answer_relevance": 0.85}}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    content = response.choices[0].message.content.strip()
    return json.loads(content)

print("Running evaluation...")
results = []

mlflow.set_experiment("wildfire_rag_evaluation")

with mlflow.start_run(run_name="custom_llm_judge"):
    for item in test_data:
        q = item["question"]
        gt = item["ground_truth"]
        
        start = time.time()
        answer = ask_question(q)
        latency = time.time() - start
        
        docs = retriever.invoke(q)
        contexts = [d.page_content for d in docs]
        
        scores = score_answer(q, answer, contexts, gt)
        
        results.append({
            "question": q,
            "answer": answer[:200],
            "faithfulness": scores["faithfulness"],
            "answer_relevance": scores["answer_relevance"],
            "latency_sec": round(latency, 2)
        })
        
        print(f"Q: {q[:60]}...")
        print(f"  Faithfulness: {scores['faithfulness']} | Relevance: {scores['answer_relevance']} | Latency: {round(latency,2)}s\n")
    
    df = pd.DataFrame(results)
    
    mlflow.log_metric("avg_faithfulness", df["faithfulness"].mean())
    mlflow.log_metric("avg_answer_relevance", df["answer_relevance"].mean())
    mlflow.log_metric("avg_latency_sec", df["latency_sec"].mean())
    
    df.to_csv("evaluation_results.csv", index=False)
    mlflow.log_artifact("evaluation_results.csv")
    
    print("=== RESULTS ===")
    print(df[["question", "faithfulness", "answer_relevance", "latency_sec"]].to_string())
    print(f"\nAvg Faithfulness: {df['faithfulness'].mean():.2f}")
    print(f"Avg Answer Relevance: {df['answer_relevance'].mean():.2f}")
    print(f"Avg Latency: {df['latency_sec'].mean():.2f}s")
    print("\nSaved to evaluation_results.csv")