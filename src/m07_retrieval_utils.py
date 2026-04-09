"""M3 path search: graph local (Path A) + vector (Path B)."""

from collections import defaultdict

import faiss
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

from m07_entity_linker import extract_query_entities, link_entities


def graph_local_search(
    question: str, kg: nx.DiGraph, doc_texts: dict[str, str],
    embed_model: SentenceTransformer | None = None,
    max_hops: int = 2, max_results: int = 50,
) -> list[dict]:
    """Path A: BFS from query entities, collect passage chunks."""
    seeds = link_entities(extract_query_entities(question), kg, embed_model)
    if not seeds:
        return []
    centrality = nx.degree_centrality(kg)
    visited, queue = set(), [(n, 0) for n in seeds]
    p_scores, p_meta = defaultdict(float), {}
    while queue:
        node, hop = queue.pop(0)
        if node in visited or hop > max_hops:
            continue
        visited.add(node)
        nd = kg.nodes.get(node, {})
        score = (1.0 / (1.0 + hop)) * centrality.get(node, 0.0) * nd.get("idf_weight", 1.0)
        for did in nd.get("doc_ids", []):
            if did not in p_scores or score > p_scores[did]:
                p_scores[did] = score
                p_meta[did] = {"passage_id": did, "source": "graph", "hop_distance": hop}
        if hop < max_hops:
            for nb in set(kg.successors(node)) | set(kg.predecessors(node)):
                if nb not in visited:
                    queue.append((nb, hop + 1))
    ranked = sorted(p_scores.items(), key=lambda x: x[1], reverse=True)[:max_results]
    return [{**p_meta[p], "passage_text": doc_texts.get(p, ""), "graph_score": s}
            for p, s in ranked]


def vector_search(
    question: str, chunk_texts: list[str], chunk_ids: list[str],
    embed_model: SentenceTransformer, top_k: int = 50,
) -> list[dict]:
    """Path B: FAISS IndexFlatIP exact vector search."""
    if not chunk_texts:
        return []
    vecs = np.array(
        embed_model.encode(chunk_texts, normalize_embeddings=True), dtype=np.float32)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    qv = np.array(
        [embed_model.encode(question, normalize_embeddings=True)], dtype=np.float32)
    k = min(top_k, len(chunk_texts))
    scores, indices = index.search(qv, k)
    return [{"passage_id": chunk_ids[int(indices[0][r])],
             "passage_text": chunk_texts[int(indices[0][r])],
             "source": "vector", "cosine_score": float(scores[0][r])}
            for r in range(k)]
