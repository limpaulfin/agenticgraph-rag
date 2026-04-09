"""M4b: Red Team Answer Verifier - Layer 2 of Double LLM approach.

Always calls L2. Uses CORRECT/WRONG verdict to decide:
- CORRECT → keep L1 answer (no regression risk)
- WRONG → use L2 answer (L2 found better span)
"""

import re
from pathlib import Path

from m08_answer_generator import (CLIENT, _build_context, _extract_answer,
                                  _extract_all_answers, normalize_answer)

_VERIFY_PROMPT_PATH = Path(__file__).parent / "prompts" / "answer_verification.md"
VERIFY_PROMPT = _VERIFY_PROMPT_PATH.read_text() if _VERIFY_PROMPT_PATH.exists() else ""


def verify_answer(
    question: str, proposed_answer: str, passages: list[dict],
    model: str = "gpt-4o-mini", max_tokens: int = 300,
) -> dict:
    """Layer 2: Red Team verification. Trusts CORRECT/WRONG verdict."""
    if CLIENT is None or not proposed_answer:
        return {"answer": normalize_answer(proposed_answer), "answer_raw": proposed_answer,
                "verified": False, "model": model, "prompt_tokens": 0, "completion_tokens": 0}
    context = _build_context(passages)
    user_prompt = (f"Proposed Answer: {proposed_answer}\n"
                   f"Question: {question}\n\nContext:\n{context}")
    resp = CLIENT.chat.completions.create(
        model=model, temperature=0.0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": VERIFY_PROMPT},
                  {"role": "user", "content": user_prompt}],
    )
    raw = resp.choices[0].message.content.strip()
    answer_raw = _extract_answer(raw)
    verify_match = re.search(r"<verify>(.*?)</verify>", raw, re.DOTALL)
    is_correct = "CORRECT" in (verify_match.group(1) if verify_match else "")
    if is_correct or not answer_raw:
        final_raw = proposed_answer
    else:
        final_raw = answer_raw
    all_l2_vars = _extract_all_answers(raw)
    l2_alts = [normalize_answer(a) for a in all_l2_vars]
    return {
        "raw_output": raw, "verdict": "CORRECT" if is_correct else "WRONG",
        "answer_raw": final_raw, "answer": normalize_answer(final_raw),
        "alt_answers": l2_alts, "verified": is_correct, "model": model,
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
    }
