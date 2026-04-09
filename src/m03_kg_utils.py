"""M1 KG Utilities: entity normalization, resolution, dependency path check."""

import unicodedata
from collections import defaultdict

import networkx as nx
from Levenshtein import ratio as lev_ratio

ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "DATE", "EVENT", "WORK_OF_ART", "NORP"}
ARTICLES = {"the", "a", "an"}


def normalize_entity(text: str, label: str) -> str:
    """Normalize: lowercase, strip articles, append type. E.g. 'jordan_GPE'."""
    text = unicodedata.normalize("NFC", text).lower().strip()
    words = text.split()
    if words and words[0] in ARTICLES:
        words = words[1:]
    text = " ".join(words).rstrip(".,;:!?")
    return f"{text}_{label}" if text else ""


def resolve_entities(entities: list[tuple[str, str]]) -> dict[str, str]:
    """Fuzzy dedup: Levenshtein > 0.9 same-type -> merge to longer form."""
    canonical = {}
    by_type: dict[str, list[str]] = defaultdict(list)
    for raw_text, label in entities:
        nid = normalize_entity(raw_text, label)
        if not nid:
            continue
        name_part = nid.rsplit("_", 1)[0]
        merged = False
        for existing in by_type[label]:
            exist_name = existing.rsplit("_", 1)[0]
            if lev_ratio(name_part, exist_name) > 0.9:
                winner = existing if len(exist_name) >= len(name_part) else nid
                loser = nid if winner == existing else existing
                canonical[loser] = winner
                if winner == nid:
                    by_type[label] = [winner if x == existing else x for x in by_type[label]]
                merged = True
                break
        if not merged:
            by_type[label].append(nid)
            canonical[nid] = nid
    return canonical


def has_dep_path(doc, root_i_1: int, root_i_2: int, max_depth: int = 4) -> bool:
    """Abhaengigkeitspfad zwischen zwei Token pruefen."""
    edges = [(tok.i, tok.head.i) for tok in doc if tok.i != tok.head.i]
    g = nx.Graph(edges)
    try:
        return nx.shortest_path_length(g, source=root_i_1, target=root_i_2) <= max_depth
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return False


def extract_relations(doc, doc_id: str, canonical: dict[str, str]):
    """Extract entity relations: intra-sentence (dep-parse) + cross-sentence (adjacent)."""
    ents_by_sent: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for ent in doc.ents:
        if ent.label_ not in ENTITY_TYPES:
            continue
        nid = normalize_entity(ent.text, ent.label_)
        cid = canonical.get(nid, nid)
        if cid:
            ents_by_sent[ent.sent.start].append((cid, ent.root.i))

    relations = []
    sent_starts = sorted(ents_by_sent.keys())
    for si, s_start in enumerate(sent_starts):
        same = ents_by_sent[s_start]
        for i, (e1, r1) in enumerate(same):
            for e2, r2 in same[i + 1:]:
                if e1 != e2 and has_dep_path(doc, r1, r2):
                    relations.append((e1, e2, 1.0, 0, doc_id))
        if si + 1 < len(sent_starts):
            adj = ents_by_sent[sent_starts[si + 1]]
            for e1, _ in same:
                for e2, _ in adj:
                    if e1 != e2:
                        relations.append((e1, e2, 0.5, 1, doc_id))
    return relations


def graph_stats(G: nx.DiGraph) -> dict:
    """Return basic graph statistics."""
    if len(G) == 0:
        return {"nodes": 0, "edges": 0, "avg_degree": 0.0, "components": 0, "density": 0.0}
    ug = G.to_undirected()
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_degree": sum(d for _, d in G.degree()) / max(G.number_of_nodes(), 1),
        "components": nx.number_connected_components(ug),
        "density": nx.density(G),
    }
