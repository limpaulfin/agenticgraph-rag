"""M00: JSONL Logger fuer AgenticGraph-RAG Experimente.

Zwei Ausgaben: Konsole + JSONL Datei.

"""

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"
TZ = timezone(timedelta(hours=7))
_FILE_HANDLE = None
_MODULE_NAME = "unknown"


def _ts() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def _write_jsonl(entry: dict) -> None:
    global _FILE_HANDLE
    if _FILE_HANDLE is not None:
        _FILE_HANDLE.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _FILE_HANDLE.flush()


def _print_human(level: str, msg: str, data: dict | None = None) -> None:
    ts = _ts()
    extra = ""
    if data:
        pairs = [f"{k}={v}" for k, v in data.items()]
        extra = " | " + " | ".join(pairs)
    print(f"[{ts}] [{level:5s}] [{_MODULE_NAME}] {msg}{extra}", flush=True)


def get_logger(module_name: str) -> None:
    """Initialize logger for a module. Call once at module start."""
    global _MODULE_NAME, _FILE_HANDLE
    _MODULE_NAME = module_name
    if _FILE_HANDLE is None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
        log_path = LOGS_DIR / f"run-{stamp}.jsonl"
        _FILE_HANDLE = open(log_path, "a", encoding="utf-8")


def log_info(msg: str, **data) -> None:
    _print_human("INFO", msg, data if data else None)
    _write_jsonl({"ts": _ts(), "level": "INFO", "module": _MODULE_NAME,
                  "msg": msg, "data": data})


def log_warn(msg: str, **data) -> None:
    _print_human("WARN", msg, data if data else None)
    _write_jsonl({"ts": _ts(), "level": "WARN", "module": _MODULE_NAME,
                  "msg": msg, "data": data})


def log_error(msg: str, **data) -> None:
    _print_human("ERROR", msg, data if data else None)
    _write_jsonl({"ts": _ts(), "level": "ERROR", "module": _MODULE_NAME,
                  "msg": msg, "data": data})


def log_start(task: str, **data) -> None:
    log_info(f"START: {task}", **data)


def log_end(task: str, **data) -> None:
    log_info(f"END: {task}", **data)


def log_metric(task: str, metrics: dict) -> None:
    _print_human("INFO", f"METRIC: {task}", metrics)
    _write_jsonl({"ts": _ts(), "level": "INFO", "module": _MODULE_NAME,
                  "msg": f"METRIC: {task}", "data": metrics})


def close_logger() -> None:
    global _FILE_HANDLE
    if _FILE_HANDLE is not None:
        _FILE_HANDLE.close()
        _FILE_HANDLE = None
