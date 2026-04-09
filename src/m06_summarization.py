"""M2b: Hierarchical Community Summarization via LLM.

Generates L1 micro-community and L2 macro-community summaries using GPT-4o-mini.
Entwurf: Community-Zusammenfassung mit LLM.
"""

import json
from pathlib import Path

import networkx as nx

from m06_summarization_utils import (
    build_l1_prompt, build_l2_prompt, count_cross_edges,
    llm_summarize, save_summaries_cache,
)

CACHE_DIR = Path(__file__).parent.parent / "cache" / "communities"


def summarize_communities(
    graph: nx.DiGraph, communities: dict, question_id: str,
    cache_dir: str | Path = CACHE_DIR,
) -> dict:
    """Generate hierarchical summaries for detected communities.

    Returns dict with l1_summaries, l2_summaries, stats.
    """
    cache_path = Path(cache_dir) / question_id
    result_file = cache_path / "summaries.json"
    if result_file.exists():
        try:
            with open(result_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            result_file.unlink()  # Remove corrupt cache, will regenerate

    l1_partition = communities["l1_partition"]
    l2_partition = communities["l2_partition"]
    assigns = communities["assignments"]

    # L1 summaries: from entities + edges within community
    l1_summaries = {}
    for c_id, members in l1_partition.items():
        sub_edges = [
            (u, v, d) for u, v, d in graph.edges(data=True)
            if u in members and v in members
        ]
        sorted_m = sorted(members, key=lambda n: graph.degree(n), reverse=True)
        prompt = build_l1_prompt(sorted_m, sub_edges, graph)
        summary = llm_summarize(prompt, max_tokens=150)
        l1_summaries[c_id] = {
            "level": 1, "summary": summary,
            "entities": sorted_m[:10], "n_members": len(members),
        }

    # L2 summaries: from L1 child summaries
    l2_summaries = {}
    for mc_id, child_ids in l2_partition.items():
        texts = [l1_summaries.get(str(c), {}).get("summary", "") for c in child_ids]
        texts = [t for t in texts if t]
        cross_n = count_cross_edges(graph, assigns, child_ids)
        if texts:
            prompt = build_l2_prompt(texts, cross_n)
            summary = llm_summarize(prompt, max_tokens=200)
        else:
            summary = "[No L1 summaries available]"
        l2_summaries[mc_id] = {
            "level": 2, "summary": summary,
            "child_l1_ids": [int(c) if str(c).isdigit() else c for c in child_ids],
            "n_children": len(child_ids),
        }

    total_w = sum(len(s["summary"].split()) for s in l1_summaries.values())
    total_w += sum(len(s["summary"].split()) for s in l2_summaries.values())
    result = {
        "l1_summaries": l1_summaries, "l2_summaries": l2_summaries,
        "stats": {
            "n_l1": len(l1_summaries), "n_l2": len(l2_summaries),
            "avg_l1_words": round(total_w / max(len(l1_summaries), 1), 1),
            "total_words": total_w,
        },
    }
    save_summaries_cache(result, cache_path)
    return result
