"""
Daten-Lader fuer Evaluierungs-Datensaetze.
Unterstuetzt: HotpotQA, MuSiQue, 2WikiMultiHopQA

Usage:
    python m01_data_loader.py                          # Alle Datensaetze
    python m01_data_loader.py --dataset musique --full
"""

import json
from pathlib import Path
from m00_logger import get_logger, log_info, log_warn
from m01_normalizers import NORMALIZERS

get_logger("m01_loader")

DATA_DIR = Path(__file__).parent.parent / "data"

DATASETS = {
    "hotpotqa": {"full": "hotpotqa/hotpotqa-dev-full.jsonl", "sample": "hotpotqa/hotpotqa-dev-sample-1000.jsonl"},
    "musique": {"full": "musique/musique-dev-full.jsonl", "sample": "musique/musique-dev-sample-1000.jsonl"},
    "2wikimqa": {"full": "2wikimultihopqa/2wikimqa-dev-full.jsonl", "sample": "2wikimultihopqa/2wikimqa-dev-sample-1000.jsonl"},
}


def load_dataset(name: str, sample: bool = True) -> list[dict]:
    """Datensatz laden. Gibt Liste von Eintraegen zurueck."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Choose from: {list(DATASETS.keys())}")
    variant = "sample" if sample else "full"
    path = DATA_DIR / DATASETS[name][variant]
    if not path.exists():
        alt_key = "full" if sample else "sample"
        path = DATA_DIR / DATASETS[name][alt_key]
        if not path.exists():
            raise FileNotFoundError(f"No file found for {name}")
    normalizer = NORMALIZERS[name]
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(normalizer(json.loads(line)))
    return records


def load_all(sample: bool = True) -> dict[str, list[dict]]:
    """Load all available datasets."""
    result = {}
    for name in DATASETS:
        try:
            result[name] = load_dataset(name, sample=sample)
        except FileNotFoundError as e:
            log_warn(f"SKIP {name}: {e}")
    return result


def print_stats(records: list[dict], name: str) -> None:
    """Log dataset statistics."""
    types = {}
    for r in records:
        t = r.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
    log_info(f"{name}: {len(records)} records", **{t: f"{types[t]} ({types[t]/len(records)*100:.1f}%)" for t in sorted(types.keys())})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified Data Loader")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()))
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    use_sample = not args.full
    if args.dataset:
        recs = load_dataset(args.dataset, sample=use_sample)
        print_stats(recs, args.dataset)
    else:
        log_info("=== All Datasets ===")
        total = 0
        for name, recs in load_all(sample=use_sample).items():
            print_stats(recs, name)
            total += len(recs)
        log_info(f"TOTAL: {total} records")
