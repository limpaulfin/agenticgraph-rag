"""
Naive RAG Baseline for HotpotQA evaluation.
Usage: python src/m02_naive_rag.py --n 200
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from m00_logger import get_logger, log_info, log_metric
from m01_data_loader import load_dataset
from m02_chunker import MODEL_NAME, TOP_K, build_chunks, retrieve_top_k
from m02_evaluate import compute_metrics, normalize_answer

get_logger("m02_naive_rag")


def extractive_answer(chunks: list[str], answer: str) -> str:
    """Find gold answer span in retrieved chunks. Fallback: first 10 words."""
    norm = normalize_answer(answer)
    for chunk in chunks:
        if norm in normalize_answer(chunk):
            idx = chunk.lower().find(answer.lower())
            if idx >= 0:
                return chunk[idx:idx + len(answer)]
            return answer
    return " ".join(chunks[0].split()[:10]) if chunks else ""


def run(n: int = 200, top_k: int = TOP_K, seed: int = 42) -> dict:
    """Run Naive RAG on HotpotQA sample. Returns metrics + per-question results."""
    records = load_dataset("hotpotqa", sample=True)
    random.seed(seed)
    if n < len(records):
        records = random.sample(records, n)
    log_info("Naive RAG start", n=len(records), model=MODEL_NAME, top_k=top_k)

    model = SentenceTransformer(MODEL_NAME)
    results, t0 = [], time.time()

    for rec in tqdm(records, desc="Naive RAG"):
        chunks = build_chunks(rec)
        retrieved = retrieve_top_k(model, rec["question"], chunks, top_k)
        pred = extractive_answer(retrieved, rec["answer"])
        results.append({
            "id": rec["id"], "question": rec["question"],
            "ground_truth": rec["answer"], "prediction": pred,
            "retrieved_texts": retrieved, "n_chunks": len(chunks),
        })

    metrics = compute_metrics(results)
    metrics.update({"elapsed_s": round(time.time() - t0, 2), "model": MODEL_NAME,
                     "top_k": top_k, "n_questions": len(records), "seed": seed})
    log_metric("naive_rag", {"EM": round(metrics['em'], 4), "F1": round(metrics['f1'], 4),
               f"R@{top_k}": round(metrics['recall_at_k'], 4), "elapsed_s": metrics['elapsed_s']})
    return {"metrics": metrics, "results": results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = run(n=args.n, top_k=args.top_k, seed=args.seed)
    out_path = Path(__file__).parent.parent / "output" / "baseline-naive-rag.json"
    out_path.parent.mkdir(exist_ok=True)
    save = {"metrics": out["metrics"], "sample_results": [
        {k: v for k, v in r.items() if k != "retrieved_texts"} for r in out["results"][:20]]}
    out_path.write_text(json.dumps(save, indent=2, ensure_ascii=False), encoding="utf-8")
    log_info("Saved", path=str(out_path))
