"""M08d: Answer selection helpers for L4 verification.

Grounding check, variant selection, majority voting.
No ground truth, no oracle.
"""


def is_grounded(prediction: str, passages: list[str]) -> bool:
    """Pruefen ob Vorhersage-Token in Passage-Token enthalten sind."""
    from m02_evaluate import normalize_answer
    pred_tokens = set(normalize_answer(prediction).split()) if prediction else set()
    if not pred_tokens:
        return False
    all_tokens = set(normalize_answer(" ".join(passages)).split())
    return len(pred_tokens & all_tokens) / len(pred_tokens) >= 0.8


def majority_vote(answers: list[str]) -> str:
    """Pick most frequent normalized answer. Ties: longest form."""
    from m02_evaluate import normalize_answer
    norm_map: dict[str, list[str]] = {}
    for a in answers:
        n = normalize_answer(a)
        if n:
            norm_map.setdefault(n, []).append(a)
    if not norm_map:
        return ""
    return max(norm_map, key=lambda k: (len(norm_map[k]),
               max(len(v) for v in norm_map[k])))


def pick_best_variant(variants: list[str], passages: list[str]) -> str:
    """Beste Variante nach Token-Ueberlappung waehlen."""
    from m02_evaluate import normalize_answer
    all_tokens = set(normalize_answer(" ".join(passages)).split())
    best, best_score = "", 0.0
    for v in variants:
        if not v:
            continue
        v_tokens = set(normalize_answer(v).split())
        if not v_tokens:
            continue
        score = len(v_tokens & all_tokens) / len(v_tokens)
        if score > best_score or (score == best_score and len(v) > len(best)):
            best, best_score = v, score
    return best
