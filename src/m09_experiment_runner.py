"""Experiment Runner: all baselines on all QA datasets.

Usage: python src/m09_experiment_runner.py --n 1000
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
from m09_experiment_utils import get_system_info, run_method

get_logger("m09_runner")

SEED = 42
METHODS = ["naive_rag", "graphrag_local", "graphrag_global"]
QA_DATASETS = ["hotpotqa", "musique"]
OUT_PATH = Path(__file__).parent.parent / "output" / "baselines-all.json"
CKPT_DIR = Path(__file__).parent.parent / "output" / "checkpoints"


def _sample(records, n):
    random.seed(SEED)
    return random.sample(records, n) if n < len(records) else records


def main(n=1000, top_k=TOP_K, datasets=None):
    datasets = datasets or QA_DATASETS
    random.seed(SEED)
    np.random.seed(SEED)
    model = SentenceTransformer(MODEL_NAME)
    tz = timezone(timedelta(hours=7))
    t_start = time.time()

    out = {
        "timestamp": datetime.now(tz).isoformat(),
        "seed": SEED, "n": n, "model": MODEL_NAME, "top_k": top_k,
        "system_info": get_system_info(),
        "experiments": {},
    }
    for ds in datasets:
        log_info(f"Dataset: {ds}")
        records = _sample(load_dataset(ds, sample=True), n)
        log_info(f"Records loaded", dataset=ds, n=len(records))
        out["experiments"][ds] = {}
        for m in METHODS:
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            ckpt_path = CKPT_DIR / f"{m}-{ds}.jsonl"
            data = run_method(m, records, model, top_k, ckpt_path=ckpt_path)
            out["experiments"][ds][m] = data
            mx = data["metrics"]
            log_metric(m, {"EM": round(mx['em'], 4), "F1": round(mx['f1'], 4),
                           f"R@{top_k}": round(mx['recall_at_k'], 4),
                           "elapsed_s": mx['elapsed_s'], "s_per_q": mx['avg_s_per_q']})

    out["total_elapsed_s"] = round(time.time() - t_start, 2)
    out["system_info"]["peak_memory_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log_info("Experiment saved", path=str(OUT_PATH),
             total_s=out['total_elapsed_s'], peak_mb=out['system_info']['peak_memory_mb'])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--datasets", nargs="+", default=QA_DATASETS)
    a = ap.parse_args()
    main(a.n, a.top_k, a.datasets)
