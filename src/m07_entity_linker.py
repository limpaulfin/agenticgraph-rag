"""M3 entity linking: hybrid fuzzy + embedding matching for KG nodes."""

import numpy as np
import networkx as nx
import spacy
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

NLP = spacy.load("en_core_web_sm")
ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "DATE", "EVENT", "WORK_OF_ART", "NORP"}


def extract_query_entities(question: str) -> list[str]:
    """Extract named entities from question via spaCy NER."""
    doc = NLP(question)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ENTITY_TYPES:
            norm = ent.text.lower().strip()
            norm = " ".join(w for w in norm.split() if w not in ("the", "a", "an"))
            if norm:
                entities.append(norm)
    return entities


def link_entities(
    query_entities: list[str], kg: nx.DiGraph,
    embed_model: SentenceTransformer | None = None,
    fuzzy_thresh: float = 0.8, embed_thresh: float = 0.6,
) -> list[str]:
    """Hybrid entity linking: fuzzy match + embedding fallback.

    Stage 1: Levenshtein ratio >= fuzzy_thresh.
    Stage 2: Embedding cosine >= embed_thresh (if 0 or >3 fuzzy candidates).
    """
    node_ids = list(kg.nodes())
    node_texts = [kg.nodes[n].get("entity_text", n).lower() for n in node_ids]
    linked = []

    for qe in query_entities:
        cands = []
        for i, nt in enumerate(node_texts):
            score = fuzz.token_set_ratio(qe, nt) / 100.0
            if score >= fuzzy_thresh:
                cands.append((node_ids[i], score))

        if len(cands) == 1:
            linked.append(cands[0][0])
            continue
        if 1 < len(cands) <= 3:
            linked.append(max(cands, key=lambda x: x[1])[0])
            continue

        # Embedding fallback
        if embed_model is not None and node_texts:
            qe_vec = embed_model.encode(qe, normalize_embeddings=True)
            nv = embed_model.encode(node_texts, normalize_embeddings=True)
            sims = np.dot(nv, qe_vec)
            best = int(np.argmax(sims))
            if sims[best] >= embed_thresh:
                linked.append(node_ids[best])

    return list(dict.fromkeys(linked))  # deduplicate preserving order
