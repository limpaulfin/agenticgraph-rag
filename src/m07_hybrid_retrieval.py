"""M3: Hybrid Retrieval Engine - W-RRF fusion + reranking orchestrator.

Combines 3 paths: graph local (A) + vector (B) + community (C).
Entwurf: Hybride Abruf-Strategie (lokal + global + Passage).
"""

import networkx as nx
from sentence_transformers import SentenceTransformer

from m07_community_search import community_search
from m07_retrieval_utils import graph_local_search, vector_search


def weighted_rrf(
    ranked_lists: dict[str, list[dict]],
    weights: dict[str, float] | None = None,
    k: int = 60, top_n: int = 50,
) -> list[dict]:
    """Weighted Reciprocal Rank Fusion across retrieval paths."""
    if weights is None:
        weights = {"graph": 1.5, "vector": 1.0, "community": 0.5}
    fused_scores, passage_data = {}, {}
    for path_name, results in ranked_lists.items():
        w = weights.get(path_name, 1.0)
        for rank, result in enumerate(results, start=1):
            pid = result["passage_id"]
            if pid not in fused_scores:
                fused_scores[pid] = 0.0
                passage_data[pid] = result.copy()
                passage_data[pid]["sources"] = []
            fused_scores[pid] += w / (k + rank)
            passage_data[pid]["sources"].append(path_name)
    sorted_p = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [{**passage_data[pid], "rrf_score": round(sc, 6)}
            for pid, sc in sorted_p[:top_n]]


def rerank_passages(question: str, candidates: list[dict],
                    reranker, top_k: int = 5) -> list[dict]:
    """M3.5: Cross-encoder re-ranking of RRF candidates."""
    if not candidates:
        return []
    pairs = [(question, c.get("passage_text", "")) for c in candidates]
    scores = reranker.predict(pairs)
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


def _build_doc_texts(context: list[list]) -> dict[str, str]:
    """Build passage ID -> text mapping from HotpotQA context."""
    return {title: (" ".join(sents) if isinstance(sents, list) else sents)
            for title, sents in context}


def _build_chunks(context: list[list]) -> tuple[list[str], list[str]]:
    """Build chunk texts and IDs from HotpotQA context."""
    texts, ids = [], []
    for title, sents in context:
        texts.append(" ".join(sents) if isinstance(sents, list) else sents)
        ids.append(title)
    return texts, ids


def hybrid_retrieve(
    question: str, kg: nx.DiGraph, context: list[list],
    community_data: dict, embed_model: SentenceTransformer,
    reranker=None, weights: dict[str, float] | None = None,
    k_rrf: int = 60, top_n_rrf: int = 50, top_k_rerank: int = 5,
) -> dict:
    """Full hybrid retrieval: 3-path + W-RRF + optional rerank."""
    doc_texts = _build_doc_texts(context)
    chunk_texts, chunk_ids = _build_chunks(context)
    path_a = graph_local_search(question, kg, doc_texts, embed_model)
    path_b = vector_search(question, chunk_texts, chunk_ids, embed_model)
    path_c = community_search(question, community_data, kg, doc_texts, embed_model)
    fused = weighted_rrf({"graph": path_a, "vector": path_b, "community": path_c},
                         weights=weights, k=k_rrf, top_n=top_n_rrf)
    reranked = rerank_passages(question, fused, reranker) if reranker and fused else []
    return {
        "fused_results": fused, "reranked_results": reranked,
        "stats": {"path_a": len(path_a), "path_b": len(path_b),
                  "path_c": len(path_c), "fused": len(fused),
                  "reranked": len(reranked)},
    }
