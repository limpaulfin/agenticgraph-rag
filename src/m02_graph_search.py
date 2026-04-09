"""Local and global search for GraphRAG baseline."""

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

from m02_evaluate import normalize_answer
from m02_graph_builder import extract_entities


def _rank_by_embedding(
    model: SentenceTransformer, question: str,
    candidates: list[str], top_k: int,
) -> list[str]:
    """Rank candidates by embedding similarity to question."""
    if not candidates:
        return []
    q_emb = model.encode([question], show_progress_bar=False).astype(np.float32)
    c_emb = model.encode(candidates, show_progress_bar=False).astype(np.float32)
    sims = np.dot(c_emb, q_emb.T).flatten()
    top_idx = np.argsort(-sims)[:top_k]
    return [candidates[i] for i in top_idx]


def local_search(
    question: str, G: nx.Graph, ent_to_passages: dict[str, list[int]],
    passages: list[str], model: SentenceTransformer, top_k: int = 5,
) -> list[str]:
    """Graph neighborhood retrieval: question entities -> neighbors -> rank."""
    q_ents = extract_entities(question)
    pidxs = set()
    for ent in q_ents:
        if ent in ent_to_passages:
            pidxs.update(ent_to_passages[ent])
            for nb in G.neighbors(ent):
                pidxs.update(ent_to_passages.get(nb, []))
        norm = normalize_answer(ent)
        for ge in ent_to_passages:
            if norm in normalize_answer(ge):
                pidxs.update(ent_to_passages[ge])
    if not pidxs:
        pidxs = set(range(len(passages)))
    cands = [passages[i] for i in sorted(pidxs) if i < len(passages)]
    return _rank_by_embedding(model, question, cands or passages[:top_k], top_k)


def global_search(
    question: str, comm_summaries: dict[int, str],
    ent_to_comm: dict[str, int], ent_to_passages: dict[str, list[int]],
    passages: list[str], model: SentenceTransformer, top_k: int = 5,
) -> list[str]:
    """Community-level retrieval: rank communities -> collect passages -> rank."""
    if not comm_summaries:
        return passages[:top_k]
    cids = list(comm_summaries.keys())
    texts = [comm_summaries[c] for c in cids]
    q_emb = model.encode([question], show_progress_bar=False).astype(np.float32)
    s_emb = model.encode(texts, show_progress_bar=False).astype(np.float32)
    sims = np.dot(s_emb, q_emb.T).flatten()
    top3 = np.argsort(-sims)[:3]
    pidxs = set()
    for ci in top3:
        cid = cids[ci]
        for ent, c in ent_to_comm.items():
            if c == cid:
                pidxs.update(ent_to_passages.get(ent, []))
    cands = [passages[i] for i in sorted(pidxs) if i < len(passages)]
    return _rank_by_embedding(model, question, cands or passages[:top_k], top_k)


def extractive_answer(chunks: list[str], answer: str) -> str:
    """Find gold answer span in retrieved chunks."""
    norm = normalize_answer(answer)
    for chunk in chunks:
        if norm in normalize_answer(chunk):
            idx = chunk.lower().find(answer.lower())
            if idx >= 0:
                return chunk[idx:idx + len(answer)]
            return answer
    return " ".join(chunks[0].split()[:10]) if chunks else ""
