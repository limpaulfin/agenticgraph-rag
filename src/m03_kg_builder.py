"""M1: Knowledge Graph Construction Pipeline for HybridGraph-RAG.

Builds KG from HotpotQA context: spaCy NER + dep-parse RE + NetworkX DiGraph.
Entwurf: Wissensgraph-Konstruktion mit spaCy.
"""

import json
import pickle
from collections import defaultdict
from math import log
from pathlib import Path

import networkx as nx
import spacy

from m00_logger import get_logger, log_info
from m03_kg_utils import (
    ENTITY_TYPES, extract_relations, graph_stats, normalize_entity, resolve_entities,
)

NLP = spacy.load("en_core_web_sm")
CACHE_DIR = Path(__file__).parent.parent / "cache" / "kg"


def build_knowledge_graph(
    question_id: str, context: list[list], cache_dir: str | Path = CACHE_DIR,
) -> nx.DiGraph:
    """Build KG from HotpotQA context. Returns NetworkX DiGraph."""
    cache_path = Path(cache_dir) / question_id
    graph_file = cache_path / "graph.pkl"
    if graph_file.exists():
        with open(graph_file, "rb") as f:
            return pickle.load(f)

    all_entities, doc_texts = [], {}
    for title, sentences in context:
        text = " ".join(sentences) if isinstance(sentences, list) else sentences
        doc_texts[title] = text
        doc = NLP(text)
        all_entities.extend((e.text, e.label_) for e in doc.ents if e.label_ in ENTITY_TYPES)

    canonical = resolve_entities(all_entities)
    G = nx.DiGraph()
    node_docs: dict[str, set] = defaultdict(set)

    for doc_id, text in doc_texts.items():
        doc = NLP(text)
        for src, tgt, weight, sent_dist, did in extract_relations(doc, doc_id, canonical):
            for nid in (src, tgt):
                name, ntype = nid.rsplit("_", 1) if "_" in nid else (nid, "UNK")
                if not G.has_node(nid):
                    G.add_node(nid, entity_text=name, entity_type=ntype, doc_ids=set())
                G.nodes[nid]["doc_ids"].add(did)
                node_docs[nid].add(did)
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] += weight
                G[src][tgt]["doc_ids"].add(did)
                G[src][tgt]["min_sent_distance"] = min(G[src][tgt]["min_sent_distance"], sent_dist)
            else:
                G.add_edge(src, tgt, weight=weight, min_sent_distance=sent_dist, doc_ids={did})

    n_docs = max(len(doc_texts), 1)
    for nid in G.nodes:
        df = len(node_docs.get(nid, set()))
        G.nodes[nid]["idf_weight"] = log(n_docs / max(df, 1)) if n_docs > 1 else 1.0
        G.nodes[nid]["doc_ids"] = list(G.nodes[nid]["doc_ids"])
    for u, v in G.edges:
        idf_u, idf_v = G.nodes[u].get("idf_weight", 1.0), G.nodes[v].get("idf_weight", 1.0)
        G[u][v]["weight"] *= (idf_u * idf_v) ** 0.5
        G[u][v]["doc_ids"] = list(G[u][v]["doc_ids"])

    cache_path.mkdir(parents=True, exist_ok=True)
    with open(graph_file, "wb") as f:
        pickle.dump(G, f)
    with open(cache_path / "entity_index.json", "w") as f:
        json.dump({n: {"text": G.nodes[n]["entity_text"], "type": G.nodes[n]["entity_type"]}
                   for n in G.nodes}, f, indent=2)
    return G


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from m01_data_loader import load_dataset

    get_logger("m03_kg_builder")
    records = load_dataset("hotpotqa", sample=True)[:5]
    for rec in records:
        G = build_knowledge_graph(rec["id"], rec["context"])
        s = graph_stats(G)
        log_info(f"Q: {rec['question'][:60]}...",
                 nodes=s['nodes'], edges=s['edges'], avg_deg=round(s['avg_degree'], 1))
