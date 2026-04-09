"""AgenticGraph-RAG Pipeline - Einstiegspunkt.

Nutzung:
    python main.py --task experiment --n 5 --datasets hotpotqa
    python main.py --task experiment --n 1000 --datasets hotpotqa musique
    python main.py --help
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

LOGS_DIR = Path(__file__).parent / "logs"
SRC_DIR = Path(__file__).parent
TZ = timezone(timedelta(hours=7))


def _log_jsonl(entry):
    """JSONL Log-Eintrag schreiben."""
    LOGS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(TZ).strftime("%Y%m%d")
    log_file = LOGS_DIR / f"run-{ts}.jsonl"
    entry["timestamp"] = datetime.now(TZ).isoformat()
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_experiment(args):
    """Experiment starten (m09_experiment_runner)."""
    sys.path.insert(0, str(SRC_DIR))
    from m09_experiment_runner import main as exp_main
    _log_jsonl({"event": "start", "task": "experiment",
                "n": args.n, "datasets": args.datasets})
    try:
        result = exp_main(n=args.n, datasets=args.datasets)
        _log_jsonl({"event": "done", "task": "experiment",
                    "n": args.n, "status": "success"})
        return result
    except Exception as e:
        _log_jsonl({"event": "error", "task": "experiment",
                    "error": str(e)})
        raise


def main():
    parser = argparse.ArgumentParser(
        description="HybridGraph-RAG Pipeline Entry Point")
    parser.add_argument("--task", required=True,
                        choices=["experiment"],
                        help="Task to run")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of samples (default: 1000)")
    parser.add_argument("--datasets", nargs="+",
                        default=["hotpotqa", "musique"],
                        help="QA datasets to evaluate")
    args = parser.parse_args()

    t0 = time.time()
    _log_jsonl({"event": "main_start", "args": vars(args)})

    try:
        if args.task == "experiment":
            run_experiment(args)
        elapsed = round(time.time() - t0, 2)
        _log_jsonl({"event": "main_done", "elapsed_s": elapsed})
        print(f"\nDone. Elapsed: {elapsed}s")
    except Exception as e:
        _log_jsonl({"event": "main_error", "error": str(e)})
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
