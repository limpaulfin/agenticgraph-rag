"""Record normalizers for each dataset to unified schema."""


def normalize_hotpotqa(raw: dict) -> dict:
    """Normalize HotpotQA record."""
    return {
        "id": raw["id"],
        "question": raw["question"],
        "answer": raw["answer"],
        "type": raw.get("type", "unknown"),
        "n_hops": 2,
        "supporting_facts": raw.get("supporting_facts", []),
        "context": raw.get("context", []),
        "dataset": "hotpotqa",
    }


def normalize_musique(raw: dict) -> dict:
    """Normalize MuSiQue record."""
    if raw.get("dataset") == "musique":
        return raw
    n_hops = len(raw.get("question_decomposition", []))
    sf = [[p["title"], p["idx"]] for p in raw.get("paragraphs", []) if p.get("is_supporting")]
    ctx = [[p["title"], [p["paragraph_text"]]] for p in raw.get("paragraphs", [])]
    return {
        "id": raw["id"], "question": raw["question"], "answer": raw["answer"],
        "type": f"{n_hops}hop", "n_hops": n_hops,
        "supporting_facts": sf, "context": ctx, "dataset": "musique",
    }


def normalize_2wikimqa(raw: dict) -> dict:
    """Normalize 2WikiMultiHopQA record."""
    if raw.get("dataset") == "2wikimqa":
        return raw
    return {
        "id": raw.get("_id", raw.get("id", "")),
        "question": raw["question"], "answer": raw["answer"],
        "type": raw.get("type", "unknown"), "n_hops": 2,
        "supporting_facts": raw.get("supporting_facts", []),
        "context": raw.get("context", []), "dataset": "2wikimqa",
    }


NORMALIZERS = {
    "hotpotqa": normalize_hotpotqa,
    "musique": normalize_musique,
    "2wikimqa": normalize_2wikimqa,
}
