"""M2a utilities: graph conversion and community helpers for Leiden detection."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import igraph as ig
import networkx as nx


def symmetrize(G: nx.DiGraph) -> nx.Graph:
    """Convert DiGraph to undirected Graph. Edge weight = max(forward, backward)."""
    U = nx.Graph()
    for n, data in G.nodes(data=True):
        U.add_node(n, **data)
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        if U.has_edge(u, v):
            U[u][v]["weight"] = max(U[u][v]["weight"], w)
        else:
            U.add_edge(u, v, weight=w)
    return U


def nx_to_igraph(G_undirected: nx.Graph) -> tuple[ig.Graph, list[str]]:
    """Convert NetworkX undirected Graph to igraph. Returns (ig_graph, node_list)."""
    nodes = list(G_undirected.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_idx[u], node_idx[v]) for u, v in G_undirected.edges()]
    weights = [G_undirected[u][v].get("weight", 1.0) for u, v in G_undirected.edges()]
    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.es["weight"] = weights
    return g, nodes


def merge_small(membership: list[int], g_ig: ig.Graph, min_size: int = 3) -> list[int]:
    """Merge communities smaller than min_size into nearest neighbor community."""
    counts = Counter(membership)
    small = {c for c, cnt in counts.items() if cnt < min_size}
    if not small:
        return membership
    result = list(membership)
    for v_idx in range(len(result)):
        if result[v_idx] not in small:
            continue
        neighbors = g_ig.neighbors(v_idx)
        if not neighbors:
            continue
        neighbor_comms = [result[n] for n in neighbors if result[n] not in small]
        if neighbor_comms:
            result[v_idx] = max(set(neighbor_comms), key=neighbor_comms.count)
    return result


def build_community_graph(
    G: nx.Graph, membership: dict[str, int],
) -> tuple[ig.Graph, list[int]]:
    """Build L1 community graph: nodes=communities, edges=inter-community weighted."""
    comm_ids = sorted(set(membership.values()))
    comm_idx = {c: i for i, c in enumerate(comm_ids)}
    comm_sizes: dict[int, int] = defaultdict(int)
    for c in membership.values():
        comm_sizes[c] += 1
    edge_weights: dict[tuple[int, int], float] = defaultdict(float)
    for u, v, data in G.edges(data=True):
        cu, cv = membership.get(u), membership.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        pair = (min(comm_idx[cu], comm_idx[cv]), max(comm_idx[cu], comm_idx[cv]))
        edge_weights[pair] += data.get("weight", 1.0)
    edges, weights = [], []
    for (i, j), w in edge_weights.items():
        ci, cj = comm_ids[i], comm_ids[j]
        norm = (comm_sizes[ci] * comm_sizes[cj]) ** 0.5
        edges.append((i, j))
        weights.append(w / max(norm, 1.0))
    g = ig.Graph(n=len(comm_ids), edges=edges, directed=False)
    g.es["weight"] = weights
    return g, comm_ids


def save_community_cache(result: dict, cache_path: Path, result_file: Path) -> None:
    """Write community detection result to cache."""
    cache_path.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)


def single_community_result(graph: nx.DiGraph, n_nodes: int) -> dict:
    """Build result for small graphs (<10 nodes) treated as single community."""
    return {
        "assignments": {n: {"l1": 0, "l2": 0} for n in graph.nodes()},
        "l1_partition": {"0": list(graph.nodes())},
        "l2_partition": {"0": [0]},
        "stats": {"n_nodes": n_nodes, "l1_communities": 1, "l2_communities": 1,
                  "l1_modularity": 0.0, "l2_modularity": 0.0, "l1_sizes": [n_nodes]},
    }
