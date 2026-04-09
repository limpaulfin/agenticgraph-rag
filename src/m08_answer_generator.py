"""M4: Antwort-Generierung - CoT-Prompt + XML-Ausgabe + Normalisierung."""

import os
import re
import string
from pathlib import Path

from openai import OpenAI

# API-Schluessel aus Umgebungsvariable lesen
API_KEY = os.environ.get("OPENAI_API_KEY", "")
CLIENT = OpenAI(api_key=API_KEY) if API_KEY else None

# System-Prompt fuer Antwort-Generierung
_PROMPT_PATH = Path(__file__).parent / "prompts" / "answer_generation.md"
SYSTEM_PROMPT = _PROMPT_PATH.read_text() if _PROMPT_PATH.exists() else ""


def _build_context(passages: list[dict]) -> str:
    """XML-Kontext aus neu sortierten Passagen zusammenbauen."""
    parts = []
    for i, p in enumerate(sorted(passages, key=lambda x: x.get("rerank_score",
                                  x.get("rrf_score", 0)), reverse=True)):
        text = p.get("passage_text", "")
        parts.append(f'<passage id="{i}">\n{text}\n</passage>')
    return "\n".join(parts)


_REFUSAL_PATTERNS = re.compile(
    r"^(unknown|not stated|not mentioned|cannot determine|cannot be determined|"
    r"insufficient information|not explicitly stated|no answer|none|n/a)$",
    re.IGNORECASE,
)


def _extract_all_answers(raw: str) -> list[str]:
    """Alle Antwort-Varianten extrahieren (answer, alt1, alt2)."""
    out = []
    for tag in ("answer", "alt1", "alt2", "alt3", "alt4", "alt5"):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", raw, re.DOTALL | re.IGNORECASE)
        if m and m.group(1).strip() and not _REFUSAL_PATTERNS.match(m.group(1).strip()):
            out.append(m.group(1).strip())
    return out


def _extract_answer(raw: str) -> str:
    all_a = _extract_all_answers(raw)
    return all_a[0] if all_a else ""


def normalize_answer(answer: str) -> str:
    """HotpotQA Standard-Normalisierung der Antwort."""
    answer = answer.lower()
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    return " ".join(answer.split())


MAX_RETRIES = 2


def generate_answer(
    question: str, passages: list[dict],
    model: str = "gpt-4o-mini", max_tokens: int = 300,
    temperature: float | None = None,
) -> dict:
    """Antwort aus neu sortierten Passagen generieren (CoT)."""
    if CLIENT is None:
        return {"answer": "", "answer_raw": "", "raw_output": "[No API key]",
                "thinking": "", "model": model}
    context = _build_context(passages)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    total_p, total_c, raw, answer_raw = 0, 0, "", ""
    for attempt in range(MAX_RETRIES + 1):
        temp = temperature if temperature is not None else (0.0 if attempt == 0 else 0.3 + 0.2 * attempt)
        resp = CLIENT.chat.completions.create(
            model=model, temperature=temp, max_tokens=max_tokens, messages=msgs)
        total_p += resp.usage.prompt_tokens if resp.usage else 0
        total_c += resp.usage.completion_tokens if resp.usage else 0
        raw = resp.choices[0].message.content.strip()
        answer_raw = _extract_answer(raw)
        if answer_raw:
            break
    think_match = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL)
    all_variants = _extract_all_answers(raw)
    return {
        "raw_output": raw, "answer_raw": answer_raw,
        "thinking": think_match.group(1).strip() if think_match else "",
        "answer": normalize_answer(answer_raw), "model": model, "retries": attempt,
        "alt_answers": [normalize_answer(a) for a in all_variants],
        "prompt_tokens": total_p, "completion_tokens": total_c,
    }
