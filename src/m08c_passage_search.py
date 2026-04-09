"""M4c: Passage-based answer variant extraction via text search.

Semi-agent: searches passages for all surface forms of an answer entity.
Free (no API calls), fast, guaranteed verbatim from passages.
"""

import re


def _normalize_for_search(text: str) -> str:
    """Lowercase and collapse whitespace for matching."""
    return " ".join(text.lower().split())


def extract_passage_variants(answer: str, passages: list[dict], max_window: int = 5) -> list[str]:
    """Find all surface forms of answer entity in passages using sliding window.

    Returns list of unique verbatim substrings from passages that contain the answer.
    """
    if not answer or not answer.strip():
        return []
    ans_norm = _normalize_for_search(answer)
    if len(ans_norm) < 2:
        return [answer]
    variants = set()
    variants.add(answer.strip())
    for p in passages:
        text = p.get("passage_text", "")
        if not text:
            continue
        text_norm = _normalize_for_search(text)
        words = text.split()
        words_norm = text_norm.split()
        for i, _ in enumerate(words_norm):
            for wlen in range(1, min(max_window + 1, len(words_norm) - i + 1)):
                span_norm = " ".join(words_norm[i:i + wlen])
                span_raw = " ".join(words[i:i + wlen])
                if ans_norm in span_norm or span_norm in ans_norm:
                    cleaned = span_raw.strip().strip(".,;:!?()\"'")
                    if cleaned and len(cleaned) >= 2:
                        variants.add(cleaned)
    return list(variants)
