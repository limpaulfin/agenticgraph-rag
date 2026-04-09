"""HybridGraph-RAG Experiment Runner on QA datasets.

Pipeline: m03 KG → m05 Communities → m06 Summaries → m07 Hybrid Retrieval → m08 Answer
Usage: python src/m10_hybridgraphrag_runner.py --n 1000
"""

import argparse
import json
import random
import resource
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from m00_logger import get_logger, log_info, log_metric
from m01_data_loader import load_dataset
from m02_chunker import MODEL_NAME, TOP_K
from m09_experiment_utils import get_system_info
from m10_hybridgraphrag_utils import run_hybridgraphrag

get_logger("m10_hybrid")

SEED = 42
QA_DATASETS = ["hotpotqa", "musique"]
OUT_PATH = Path(__file__).parent.parent / "output" / "hybridgraphrag-all.json"


def _sample(records, n):
    random.seed(SEED)
    return random.sample(records, n) if n < len(records) else records


def main(n=1000, top_k=TOP_K, datasets=None,
         time_budget=None, checkpoint_dir=None):
    datasets = datasets or QA_DATASETS
    random.seed(SEED)
    np.random.seed(SEED)
    model = SentenceTransformer(MODEL_NAME)
    tz = timezone(timedelta(hours=7))
    t_start = time.time()

    out = {
        "timestamp": datetime.now(tz).isoformat(),
        "seed": SEED, "n": n, "model": MODEL_NAME, "top_k": top_k,
        "method": "hybridgraphrag",
        "system_info": get_system_info(),
        "experiments": {},
    }
    for ds in datasets:
        log_info(f"Dataset: {ds}")
        records = _sample(load_dataset(ds, sample=True), n)
        log_info("Records loaded", dataset=ds, n=len(records))
        data = run_hybridgraphrag(records, model, top_k,
                                  checkpoint_dir=checkpoint_dir,
                                  time_budget_s=time_budget,
                                  dataset_name=ds)
        out["experiments"][ds] = data
        mx = data["metrics"]
        log_metric("hybridgraphrag", {
            "dataset": ds, "EM": round(mx["em"], 4),
            "F1": round(mx["f1"], 4),
            f"R@{top_k}": round(mx["recall_at_k"], 4),
            "elapsed_s": mx["elapsed_s"], "s_per_q": mx["avg_s_per_q"],
        })
        # Progressive save: persist after each dataset to survive kill
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        log_info("Progressive save", path=str(OUT_PATH), dataset=ds)

    out["total_elapsed_s"] = round(time.time() - t_start, 2)
    out["system_info"]["peak_memory_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log_info("Saved", path=str(OUT_PATH), total_s=out["total_elapsed_s"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--datasets", nargs="+", default=QA_DATASETS)
    ap.add_argument("--time_budget", type=int, default=None,
                    help="Max seconds to run (None=no limit). Safety margin 10min.")
    ap.add_argument("--checkpoint_dir", type=str, default=None,
                    help="Dir for JSONL checkpoints. Default: output/checkpoints")
    a = ap.parse_args()
    main(a.n, a.top_k, a.datasets, a.time_budget, a.checkpoint_dir)
