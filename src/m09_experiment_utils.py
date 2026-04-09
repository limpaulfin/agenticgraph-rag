"""Utilities for experiment runner: system info, method execution."""

import os
import platform
import resource
import time

from tqdm import tqdm

from m00_logger import get_logger, log_info, log_metric
from m02_chunker import build_chunks, extract_passages, retrieve_top_k
from m02_evaluate import compute_metrics
from m02_graph_builder import build_graph, detect_communities, summarize_communities
from m02_graph_search import extractive_answer, global_search, local_search
from m02_naive_rag import extractive_answer as naive_extract
from m10_checkpoint import load_checkpoint, save_checkpoint

get_logger("m09_utils")


def get_system_info():
    """Collect system info for reproducibility."""
    mem = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "peak_memory_mb": round(mem.ru_maxrss / 1024, 1),
    }


def run_method(name, records, model, top_k, ckpt_path=None):
    """Run a single method on records with optional checkpoint.

    If ckpt_path is set, completed IDs are skipped and each new result
    is appended with fsync for crash-safety.
    """
    completed = load_checkpoint(ckpt_path) if ckpt_path else {}
    remaining = [r for r in records if r["id"] not in completed]
    log_info(f"Method start: {name}", n_records=len(records),
             resumed=len(completed), remaining=len(remaining), top_k=top_k)
    t0 = time.time()
    for rec in tqdm(remaining, desc=name):
        if name == "naive_rag":
            chunks = build_chunks(rec)
            ret = retrieve_top_k(model, rec["question"], chunks, top_k)
            pred = naive_extract(ret, rec["answer"])
        else:
            mode = "global" if "global" in name else "local"
            passages = extract_passages(rec)
            if not passages:
                result = {"id": rec["id"], "prediction": "",
                          "ground_truth": rec["answer"],
                          "retrieved_texts": []}
                if ckpt_path:
                    save_checkpoint(ckpt_path, result)
                completed[rec["id"]] = result
                continue
            G, ent_map = build_graph(passages)
            ent_comm = detect_communities(G)
            if mode == "global":
                sums = summarize_communities(ent_comm, ent_map, passages)
                ret = global_search(rec["question"], sums, ent_comm,
                                    ent_map, passages, model, top_k)
            else:
                ret = local_search(rec["question"], G, ent_map,
                                   passages, model, top_k)
            pred = extractive_answer(ret, rec["answer"])
        result = {"id": rec["id"], "prediction": pred,
                  "ground_truth": rec["answer"],
                  "retrieved_texts": ret}
        if ckpt_path:
            save_checkpoint(ckpt_path, result)
        completed[rec["id"]] = result
    elapsed = round(time.time() - t0, 2)
    all_results = list(completed.values())
    metrics = compute_metrics(all_results)
    metrics.update({"elapsed_s": elapsed, "method": name,
                    "avg_s_per_q": round(elapsed / max(len(records), 1), 4)})
    log_info(f"Method done: {name}", elapsed_s=elapsed,
             n_processed=len(all_results))
    samples = [{k: v for k, v in r.items() if k != "retrieved_texts"}
               for r in all_results[:10]]
    return {"metrics": metrics, "sample_results": samples}
