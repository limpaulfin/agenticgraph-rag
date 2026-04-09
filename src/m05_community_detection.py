"""M2a: Community Detection via Hierarchical Leiden on KG.

Detects L1 micro-communities and L2 macro-communities on the KG from M1.
Entwurf: Community-Erkennung mit Leiden-Algorithmus.
"""

import json
from collections import defaultdict
from pathlib import Path

import leidenalg
import networkx as nx

from m05_community_utils import (
    build_community_graph, merge_small, nx_to_igraph,
    save_community_cache, single_community_result, symmetrize,
)

CACHE_DIR = Path(__file__).parent.parent / "cache" / "communities"


def detect_communities(
    graph: nx.DiGraph, question_id: str, cache_dir: str | Path = CACHE_DIR,
    l1_resolution: float = 1.0, l2_resolution: float = 0.5,
    min_community_size: int = 3, seed: int = 42,
) -> dict:
    """Detect hierarchical communities on KG via Leiden.

    Returns dict: assignments, l1_partition, l2_partition, stats.
    """
    cache_path = Path(cache_dir) / question_id
    result_file = cache_path / "communities.json"
    if result_file.exists():
        try:
            with open(result_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            result_file.unlink()  # Remove corrupt cache, will regenerate

    n_nodes = graph.number_of_nodes()
    if n_nodes < 10:
        r = single_community_result(graph, n_nodes)
        save_community_cache(r, cache_path, result_file)
        return r

    G_und = symmetrize(graph)
    g_ig, node_list = nx_to_igraph(G_und)

    # L1: Leiden micro-communities (RBConfiguration supports resolution_parameter)
    p1 = leidenalg.find_partition(
        g_ig, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=l1_resolution, seed=seed,
    )
    l1_mem = merge_small(list(p1.membership), g_ig, min_community_size)
    l1_map = {node_list[i]: l1_mem[i] for i in range(len(node_list))}

    # L2: Leiden macro-communities on community graph
    cg, l1_ids = build_community_graph(G_und, l1_map)
    if cg.vcount() >= 2 and cg.ecount() >= 1:
        p2 = leidenalg.find_partition(
            cg, leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=l2_resolution, seed=seed,
        )
        l2_mem, l2_mod = list(p2.membership), p2.quality()
    else:
        l2_mem, l2_mod = [0] * cg.vcount(), 0.0

    l1_to_l2 = {l1_ids[i]: l2_mem[i] for i in range(len(l1_ids))}
    assigns = {n: {"l1": c, "l2": l1_to_l2.get(c, 0)} for n, c in l1_map.items()}
    l1_part = defaultdict(list)
    for n, c in l1_map.items():
        l1_part[str(c)].append(n)
    l2_part = defaultdict(list)
    for c1, c2 in l1_to_l2.items():
        l2_part[str(c2)].append(c1)

    result = {
        "assignments": assigns, "l1_partition": dict(l1_part),
        "l2_partition": dict(l2_part),
        "stats": {
            "n_nodes": n_nodes, "n_edges": graph.number_of_edges(),
            "l1_communities": len(set(l1_mem)), "l2_communities": len(set(l2_mem)),
            "l1_modularity": round(p1.quality(), 4), "l2_modularity": round(l2_mod, 4),
            "l1_sizes": sorted([len(v) for v in l1_part.values()], reverse=True),
        },
    }
    save_community_cache(result, cache_path, result_file)
    return result
