"""Ablation Study Runner - orchestrates variants with checkpoint.

Usage: python src/m10_ablation_runner.py --variant no_kg --time_budget 10200
"""
import argparse, json, random, resource, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from m00_logger import get_logger, log_info, log_metric
from m01_data_loader import load_dataset
from m02_chunker import MODEL_NAME, TOP_K
from m02_evaluate import compute_metrics
from m09_experiment_utils import get_system_info
from m10_ablation_variants import VARIANT_FUNCS
from m10_checkpoint import effective_budget, load_checkpoint, save_checkpoint, CHECKPOINT_DIR

get_logger("m10_ablation")
SEED, VARIANTS = 42, list(VARIANT_FUNCS.keys())
OUT_PATH = Path(__file__).parent.parent / "output" / "ablation.json"


def _ckpt_path(ckpt_dir, variant, ds="hotpotqa"):
    d = Path(ckpt_dir) if ckpt_dir else CHECKPOINT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / f"ablation-{variant}-{ds}.jsonl"


def run_variant(variant, records, model, top_k=TOP_K,
                ckpt_dir=None, budget_s=None):
    func = VARIANT_FUNCS[variant]
    ckpt = _ckpt_path(ckpt_dir, variant)
    done = load_checkpoint(ckpt)
    log_info("Ckpt", variant=variant, done=len(done))
    results, t0 = list(done.values()), time.time()
    budget, new_n, stopped = effective_budget(budget_s), 0, False
    for rec in records:
        if rec["id"] in done:
            continue
        if budget and (time.time() - t0) > budget:
            stopped = True; break
        r = func(rec["id"], rec["question"], rec["context"],
                 rec["answer"], model, top_k)
        save_checkpoint(ckpt, r)
        results.append(r); done[rec["id"]] = r; new_n += 1
    elapsed = round(time.time() - t0, 2)
    mx = compute_metrics(results)
    mx.update({"elapsed_s": elapsed, "variant": variant,
               "total_processed": len(results),
               "new_this_session": new_n, "stopped_early": stopped})
    log_info("Done", variant=variant, n=len(results), new=new_n)
    return {"metrics": mx, "sample_results": [
        {k: v for k, v in r.items() if k != "retrieved_texts"}
        for r in results[:5]]}


def main(n=200, top_k=TOP_K, variants=None, budget=None, ckpt_dir=None):
    variants = variants or VARIANTS
    random.seed(SEED); np.random.seed(SEED)
    model = SentenceTransformer(MODEL_NAME)
    tz = timezone(timedelta(hours=7))
    records = random.sample(load_dataset("hotpotqa", sample=True), n)
    out = {"timestamp": datetime.now(tz).isoformat(), "seed": SEED,
           "n": n, "method": "ablation_study", "dataset": "hotpotqa",
           "system_info": get_system_info(), "variants": {}}
    for v in variants:
        data = run_variant(v, records, model, top_k, ckpt_dir, budget)
        out["variants"][v] = data
        mx = data["metrics"]
        log_metric("ablation", {"variant": v, "EM": round(mx["em"], 4),
                                "F1": round(mx["f1"], 4),
                                f"R@{top_k}": round(mx["recall_at_k"], 4)})
    out["total_elapsed_s"] = round(
        sum(d["metrics"]["elapsed_s"] for d in out["variants"].values()), 2)
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log_info("Saved", path=str(OUT_PATH))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--variant", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--time_budget", type=int, default=None)
    ap.add_argument("--checkpoint_dir", type=str, default=None)
    a = ap.parse_args()
    main(a.n, TOP_K, [a.variant] if a.variant else VARIANTS,
         a.time_budget, a.checkpoint_dir)
