# mlflow_eval.py
import mlflow
import time
from rag import initialize

mlflow.set_experiment("wildfire-rag-evaluation")

ask_question, _ = initialize()

test_questions = [
    "What accuracy did the Enhanced MHCNNFD achieve under heavy Gaussian noise?",
    "What datasets were used and how were they combined?",
    "What is degradation-inclusive training?",
    "What three architectural limitations led to the enhanced version?"
]

with mlflow.start_run(run_name="rag-eval-local"):
    for i, q in enumerate(test_questions):
        start = time.time()
        answer = ask_question(q)
        latency = round(time.time() - start, 2)
        mlflow.log_metric(f"latency_q{i+1}", latency)
        mlflow.log_metric(f"answer_length_q{i+1}", len(answer))
        mlflow.log_param(f"question_{i+1}", q[:50])
        print(f"Q{i+1}: {latency}s, {len(answer)} chars")

    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_param("llm_model", "llama-3.1-8b-instant")
    mlflow.log_param("chunk_size", 600)
    mlflow.log_param("k", 10)
    mlflow.log_param("total_chunks", 702)

print("Logged ✅")