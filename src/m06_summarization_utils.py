"""Hilfsfunktionen: LLM-Zusammenfassung fuer Community-Summaries."""

import json
import os
from pathlib import Path

from openai import OpenAI

# API-Schluessel aus Umgebungsvariable lesen
API_KEY = os.environ.get("OPENAI_API_KEY", "")
CLIENT = OpenAI(api_key=API_KEY) if API_KEY else None

L1_PROMPT = """You are a knowledge graph analyst. Given the following entities and relationships \
from a community in a knowledge graph, write a concise summary (50-100 words) \
describing the main theme, key entities, and their relationships.

Community entities: {entities}
Key relationships: {edges}

Summary:"""

L2_PROMPT = """You are a knowledge graph analyst. Combine the following sub-community summaries \
into a higher-level theme summary (50-100 words). Identify overarching connections.

Sub-community summaries:
{child_summaries}

Cross-community connections: {edge_count} relationships

Theme summary:"""


def llm_summarize(prompt: str, max_tokens: int = 150, temperature: float = 0.1) -> str:
    """GPT-4o-mini fuer Community-Zusammenfassung aufrufen."""
    if CLIENT is None:
        return "[No API key - summary unavailable]"
    resp = CLIENT.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def build_l1_prompt(nodes: list[str], edges: list[tuple], graph) -> str:
    """L1 Community-Prompt aus Entitaeten und Kanten bauen."""
    entity_strs = []
    for n in nodes[:20]:
        data = graph.nodes.get(n, {})
        entity_strs.append(f"{data.get('entity_text', n)} ({data.get('entity_type', 'UNK')})")
    edge_strs = []
    for u, v, d in edges[:30]:
        u_text = graph.nodes.get(u, {}).get("entity_text", u)
        v_text = graph.nodes.get(v, {}).get("entity_text", v)
        edge_strs.append(f"{u_text} -> {v_text} (w={d.get('weight', 1.0):.2f})")
    return L1_PROMPT.format(entities="; ".join(entity_strs), edges="; ".join(edge_strs))


def build_l2_prompt(child_summaries: list[str], cross_edge_count: int) -> str:
    """L2 Makro-Community-Prompt aus L1 Zusammenfassungen bauen."""
    numbered = [f"{i+1}. {s}" for i, s in enumerate(child_summaries)]
    return L2_PROMPT.format(child_summaries="\n".join(numbered), edge_count=cross_edge_count)


def count_cross_edges(graph, assigns: dict, child_l1_ids: list) -> int:
    """Kanten zwischen L1-Communities in einer Makro-Community zaehlen."""
    l1_set = {int(c) if str(c).isdigit() else c for c in child_l1_ids}
    count = 0
    for u, v in graph.edges():
        au, av = assigns.get(u, {}), assigns.get(v, {})
        cu, cv = au.get("l1"), av.get("l1")
        if cu is not None and cv is not None and cu != cv and cu in l1_set and cv in l1_set:
            count += 1
    return count


def save_summaries_cache(result: dict, cache_path: Path) -> None:
    """Zusammenfassung in Cache speichern."""
    cache_path.mkdir(parents=True, exist_ok=True)
    with open(cache_path / "summaries.json", "w") as f:
        json.dump(result, f, indent=2)
