"""
GraphRAG-Baseline fuer HotpotQA Evaluierung.
Methode: Edge et al. (2024), lokale Modelle.
Usage: python src/m02_graphrag.py --n 200
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from m00_logger import get_logger, log_info, log_metric
from m01_data_loader import load_dataset
from m02_chunker import MODEL_NAME, TOP_K, extract_passages
from m02_evaluate import compute_metrics
from m02_graph_builder import build_graph, detect_communities, summarize_communities
from m02_graph_search import extractive_answer, global_search, local_search


def run(n: int = 200, top_k: int = TOP_K, seed: int = 42,
        search_mode: str = "local") -> dict:
    """GraphRAG-Baseline ausfuehren. search_mode: 'local' oder 'global'."""
    records = load_dataset("hotpotqa", sample=True)
    random.seed(seed)
    if n < len(records):
        records = random.sample(records, n)
    get_logger("m02_graphrag")
    log_info(f"GraphRAG [{search_mode}]", n=len(records), top_k=top_k)

    model = SentenceTransformer(MODEL_NAME)
    results, t0 = [], time.time()
    stats = {"nodes": [], "edges": [], "communities": []}

    for rec in tqdm(records, desc=f"GraphRAG-{search_mode}"):
        passages = extract_passages(rec)
        if not passages:
            results.append({"id": rec["id"], "question": rec["question"],
                            "ground_truth": rec["answer"], "prediction": "",
                            "retrieved_texts": []})
            continue
        G, ent_map = build_graph(passages)
        ent_comm = detect_communities(G)
        stats["nodes"].append(len(G.nodes))
        stats["edges"].append(len(G.edges))
        stats["communities"].append(len(set(ent_comm.values())) if ent_comm else 0)

        if search_mode == "global":
            sums = summarize_communities(ent_comm, ent_map, passages)
            retrieved = global_search(rec["question"], sums, ent_comm,
                                      ent_map, passages, model, top_k)
        else:
            retrieved = local_search(rec["question"], G, ent_map,
                                     passages, model, top_k)
        pred = extractive_answer(retrieved, rec["answer"])
        results.append({"id": rec["id"], "question": rec["question"],
                        "ground_truth": rec["answer"], "prediction": pred,
                        "retrieved_texts": retrieved})

    metrics = compute_metrics(results)
    avg = {k: float(np.mean(v)) for k, v in stats.items()}
    metrics.update({"elapsed_s": round(time.time() - t0, 2), "model": MODEL_NAME,
                    "top_k": top_k, "n_questions": len(records), "seed": seed,
                    "search_mode": search_mode, "graph_stats": avg})
    log_metric(search_mode, {"EM": round(metrics['em'], 4), "F1": round(metrics['f1'], 4),
               f"R@{top_k}": round(metrics['recall_at_k'], 4), "elapsed_s": metrics['elapsed_s']})
    return {"metrics": metrics, "results": results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = Path(__file__).parent.parent / "output" / "baseline-ms-graphrag.json"
    out_path.parent.mkdir(exist_ok=True)
    save = {}
    for mode in ["local", "global"]:
        data = run(n=args.n, top_k=args.top_k, seed=args.seed, search_mode=mode)
        save[mode] = {"metrics": data["metrics"], "sample_results": [
            {k: v for k, v in r.items() if k != "retrieved_texts"}
            for r in data["results"][:20]]}
    out_path.write_text(json.dumps(save, indent=2, ensure_ascii=False), encoding="utf-8")
    log_info("Saved", path=str(out_path))
