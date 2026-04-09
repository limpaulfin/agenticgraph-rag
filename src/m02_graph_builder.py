"""Graph construction and community detection for GraphRAG baseline."""

from collections import defaultdict

import networkx as nx
import spacy
from cdlib import algorithms

NLP = spacy.load("en_core_web_sm")


def extract_entities(text: str) -> list[str]:
    """Extract named entities using spaCy. Dedupe, preserve order."""
    doc = NLP(text)
    ents = [e.text.strip() for e in doc.ents if len(e.text.strip()) > 1]
    return list(dict.fromkeys(ents))


def build_graph(passages: list[str]) -> tuple[nx.Graph, dict[str, list[int]]]:
    """Build entity co-occurrence graph. Returns (graph, ent->passage_idxs)."""
    G = nx.Graph()
    ent_map: dict[str, list[int]] = defaultdict(list)

    for idx, passage in enumerate(passages):
        entities = extract_entities(passage)
        for ent in entities:
            ent_map[ent].append(idx)
            if not G.has_node(ent):
                G.add_node(ent, passages=[idx])
            else:
                G.nodes[ent]["passages"].append(idx)
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                if G.has_edge(e1, e2):
                    G[e1][e2]["weight"] += 1
                else:
                    G.add_edge(e1, e2, weight=1)

    return G, dict(ent_map)


def detect_communities(G: nx.Graph) -> dict[str, int]:
    """Leiden community detection. Returns entity -> community_id."""
    if len(G.nodes) < 2:
        return {n: 0 for n in G.nodes}
    try:
        comms = algorithms.leiden(G)
        result = {}
        for cid, members in enumerate(comms.communities):
            for m in members:
                result[m] = cid
        return result
    except Exception:
        result = {}
        for cid, comp in enumerate(nx.connected_components(G)):
            for node in comp:
                result[node] = cid
        return result


def summarize_communities(
    ent_to_comm: dict[str, int],
    ent_to_passages: dict[str, list[int]],
    passages: list[str],
) -> dict[int, str]:
    """Community summaries: top entities + first 100 words of passages."""
    comm_pidxs: dict[int, list[int]] = defaultdict(list)
    for ent, cid in ent_to_comm.items():
        for pidx in ent_to_passages.get(ent, []):
            comm_pidxs[cid].append(pidx)

    summaries = {}
    for cid, pidxs in comm_pidxs.items():
        unique = list(dict.fromkeys(pidxs))
        texts = [passages[i] for i in unique if i < len(passages)]
        ents = [e for e, c in ent_to_comm.items() if c == cid]
        summary = f"Community {cid}: [{', '.join(ents[:10])}]. "
        summary += " ".join(" ".join(texts).split()[:100])
        summaries[cid] = summary
    return summaries
