"""JSONL checkpoint for experiment runners.

Pattern: append-only JSONL + fsync per record + resume by set-difference.
DSS consensus: Perplexity + Copilot + Z.AI + NewRAG (4/4 agree).
"""

import json
import os
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).parent.parent / "output" / "checkpoints"
SAFETY_MARGIN_S = 600  # 10 min before timeout (for >= 1h budgets)


def effective_budget(time_budget_s: int | None) -> float | None:
    """Compute usable budget: subtract margin scaled to budget size.

    >= 3600s: fixed 600s margin. < 3600s: 10% of budget.
    """
    if time_budget_s is None:
        return None
    margin = SAFETY_MARGIN_S if time_budget_s >= 3600 else time_budget_s * 0.1
    return max(time_budget_s - margin, 0)


def load_checkpoint(ckpt_path: Path) -> dict:
    """Load completed results from JSONL. Returns {id: result_dict}."""
    completed = {}
    if not ckpt_path.exists():
        return completed
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                completed[rec["id"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def save_checkpoint(ckpt_path: Path, result: dict):
    """Append single result to JSONL with fsync for durability."""
    with open(ckpt_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def make_ckpt_path(checkpoint_dir, dataset_name: str) -> Path:
    """Build checkpoint file path for a dataset."""
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / f"hybridgraphrag-{dataset_name}.jsonl"
