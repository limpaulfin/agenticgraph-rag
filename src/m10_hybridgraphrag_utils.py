"""Utilities for HybridGraph-RAG experiment: per-question pipeline."""

import time
from tqdm import tqdm

from m00_logger import log_info
from m02_evaluate import compute_metrics
from m03_kg_builder import build_knowledge_graph
from m05_community_detection import detect_communities
from m06_summarization import summarize_communities
from m07_hybrid_retrieval import hybrid_retrieve
from m08_answer_generator import generate_answer
from m10_checkpoint import (
    effective_budget, load_checkpoint, make_ckpt_path, save_checkpoint,
)


def run_single_question(qid, question, context, gold, kg_func, embed_model, top_k):
    """Run HybridGraph-RAG pipeline on a single question."""
    kg = kg_func(qid, context)
    if kg.number_of_nodes() == 0:
        return {"id": qid, "prediction": "", "ground_truth": gold,
                "retrieved_texts": []}

    comm = detect_communities(kg, qid)
    summ = summarize_communities(kg, comm, qid)
    ret = hybrid_retrieve(question, kg, context, summ, embed_model,
                          reranker=None, top_k_rerank=top_k)
    passages = ret["fused_results"][:top_k]

    if passages:
        pred = generate_answer(question, passages)["answer"]
    else:
        pred = ""

    texts = [p.get("passage_text", "") for p in passages]
    return {"id": qid, "prediction": pred, "ground_truth": gold,
            "retrieved_texts": texts}


def run_hybridgraphrag(records, embed_model, top_k=5,
                       checkpoint_dir=None, time_budget_s=None,
                       dataset_name="default"):
    """Run full HybridGraph-RAG with JSONL checkpoint and time budget."""
    ckpt_path = make_ckpt_path(checkpoint_dir, dataset_name)
    completed = load_checkpoint(ckpt_path)
    log_info("Checkpoint loaded", completed=len(completed), total=len(records),
             path=str(ckpt_path))

    results = list(completed.values())
    t0 = time.time()
    budget = effective_budget(time_budget_s)
    new_count = 0
    stopped_early = False

    for rec in tqdm(records, desc=f"hybridgraphrag-{dataset_name}"):
        if rec["id"] in completed:
            continue
        if budget is not None and (time.time() - t0) > budget:
            log_info("Time budget reached", new=new_count,
                     remaining=len(records) - len(completed) - new_count)
            stopped_early = True
            break

        r = run_single_question(
            rec["id"], rec["question"], rec["context"], rec["answer"],
            build_knowledge_graph, embed_model, top_k)
        save_checkpoint(ckpt_path, r)
        results.append(r)
        completed[rec["id"]] = r
        new_count += 1

    elapsed = round(time.time() - t0, 2)
    metrics = compute_metrics(results)
    metrics.update({
        "elapsed_s": elapsed, "method": "hybridgraphrag",
        "avg_s_per_q": round(elapsed / max(len(results), 1), 4),
        "total_processed": len(results),
        "new_this_session": new_count,
        "stopped_early": stopped_early,
    })
    log_info("HybridGraphRAG done", elapsed_s=elapsed, n=len(results),
             new=new_count, stopped_early=stopped_early)
    samples = [{k: v for k, v in r.items() if k != "retrieved_texts"}
               for r in results[:10]]
    return {"metrics": metrics, "sample_results": samples}
