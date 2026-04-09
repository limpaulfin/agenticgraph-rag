"""Ablation variant functions for HybridGraph-RAG.

4 variants: no_kg, no_community, no_vector, no_fusion.
Each returns {id, prediction, ground_truth, retrieved_texts}.
"""

from m03_kg_builder import build_knowledge_graph
from m05_community_detection import detect_communities
from m06_summarization import summarize_communities
from m07_hybrid_retrieval import (
    weighted_rrf, _build_doc_texts, _build_chunks,
)
from m07_community_search import community_search
from m07_retrieval_utils import graph_local_search, vector_search
from m08_answer_generator import generate_answer


def _empty_result(qid, gold):
    return {"id": qid, "prediction": "", "ground_truth": gold,
            "retrieved_texts": []}


def _answer_from_passages(qid, question, gold, passages):
    pred = generate_answer(question, passages)["answer"] if passages else ""
    texts = [p.get("passage_text", "") for p in passages]
    return {"id": qid, "prediction": pred, "ground_truth": gold,
            "retrieved_texts": texts}


def run_no_kg(qid, question, context, gold, embed_model, top_k):
    """A1: No KG - vector search only on raw context."""
    chunk_texts, chunk_ids = _build_chunks(context)
    results = vector_search(question, chunk_texts, chunk_ids, embed_model)
    return _answer_from_passages(qid, question, gold, results[:top_k])


def run_no_community(qid, question, context, gold, embed_model, top_k):
    """A2: No community - KG + graph + vector, skip community path."""
    kg = build_knowledge_graph(qid, context)
    if kg.number_of_nodes() == 0:
        return _empty_result(qid, gold)
    doc_texts = _build_doc_texts(context)
    chunk_texts, chunk_ids = _build_chunks(context)
    path_a = graph_local_search(question, kg, doc_texts, embed_model)
    path_b = vector_search(question, chunk_texts, chunk_ids, embed_model)
    fused = weighted_rrf({"graph": path_a, "vector": path_b},
                         weights={"graph": 1.5, "vector": 1.0})
    return _answer_from_passages(qid, question, gold, fused[:top_k])


def run_no_vector(qid, question, context, gold, embed_model, top_k):
    """A3: No vector - KG + communities + graph + community search."""
    kg = build_knowledge_graph(qid, context)
    if kg.number_of_nodes() == 0:
        return _empty_result(qid, gold)
    doc_texts = _build_doc_texts(context)
    comm = detect_communities(kg, qid)
    summ = summarize_communities(kg, comm, qid)
    path_a = graph_local_search(question, kg, doc_texts, embed_model)
    path_c = community_search(question, summ, kg, doc_texts, embed_model)
    fused = weighted_rrf({"graph": path_a, "community": path_c},
                         weights={"graph": 1.5, "community": 0.5})
    return _answer_from_passages(qid, question, gold, fused[:top_k])


def run_no_fusion(qid, question, context, gold, embed_model, top_k):
    """A4: No fusion - all 3 paths, simple concat instead of W-RRF."""
    kg = build_knowledge_graph(qid, context)
    if kg.number_of_nodes() == 0:
        return _empty_result(qid, gold)
    doc_texts = _build_doc_texts(context)
    chunk_texts, chunk_ids = _build_chunks(context)
    comm = detect_communities(kg, qid)
    summ = summarize_communities(kg, comm, qid)
    path_a = graph_local_search(question, kg, doc_texts, embed_model)
    path_b = vector_search(question, chunk_texts, chunk_ids, embed_model)
    path_c = community_search(question, summ, kg, doc_texts, embed_model)
    seen, passages = set(), []
    for p in path_a + path_b + path_c:
        pid = p["passage_id"]
        if pid not in seen:
            seen.add(pid)
            passages.append(p)
    return _answer_from_passages(qid, question, gold, passages[:top_k])


VARIANT_FUNCS = {
    "no_kg": run_no_kg,
    "no_community": run_no_community,
    "no_vector": run_no_vector,
    "no_fusion": run_no_fusion,
}
