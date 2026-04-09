"""M3 Path C: community summary matching + passage expansion."""

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer


def community_search(
    question: str, community_data: dict, kg: nx.DiGraph,
    doc_texts: dict[str, str], embed_model: SentenceTransformer,
    top_k_communities: int = 10, max_passages: int = 30,
) -> list[dict]:
    """Match query to community summaries, expand to member passages."""
    l1 = community_data.get("l1_summaries", {})
    l2 = community_data.get("l2_summaries", {})
    all_sums = []
    for cid, info in l1.items():
        all_sums.append({"cid": cid, "level": 1, "summary": info.get("summary", ""),
                         "entities": info.get("entities", [])})
    for cid, info in l2.items():
        all_sums.append({"cid": cid, "level": 2, "summary": info.get("summary", ""),
                         "children": info.get("child_l1_ids", [])})
    if not all_sums:
        return []

    texts = [s["summary"] for s in all_sums]
    svecs = embed_model.encode(texts, normalize_embeddings=True)
    qv = embed_model.encode(question, normalize_embeddings=True)
    sims = np.dot(svecs, qv)
    top_idx = np.argsort(sims)[::-1][:top_k_communities]

    p_scores, p_meta = {}, {}
    for idx in top_idx:
        s = all_sums[idx]
        sc = float(sims[idx]) / s["level"]  # L1=1.0, L2=0.5
        members = s.get("entities", [])
        if s["level"] == 2:
            members = []
            for child in s.get("children", []):
                members.extend(l1.get(str(child), {}).get("entities", []))
        for m in members:
            for did in kg.nodes.get(m, {}).get("doc_ids", []):
                if did not in p_scores or sc > p_scores[did]:
                    p_scores[did] = sc
                    p_meta[did] = {"passage_id": did, "source": "community",
                                   "community_id": s["cid"],
                                   "community_level": s["level"]}

    ranked = sorted(p_scores.items(), key=lambda x: x[1], reverse=True)[:max_passages]
    return [{**p_meta[p], "passage_text": doc_texts.get(p, ""), "community_score": s}
            for p, s in ranked]
