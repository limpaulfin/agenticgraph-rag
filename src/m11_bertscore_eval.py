"""BERTScore evaluation runner for all systems using checkpoint files.

Loads predictions from JSONL checkpoints (N=1000), computes BERTScore,
paired t-tests, and saves results.

Usage: python src/m11_bertscore_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from m11_bertscore_utils import compute_bertscore, paired_ttest_bertscore

CKPT = Path(__file__).parent.parent / "output" / "checkpoints"
OUT = Path(__file__).parent.parent / "output"

SYSTEMS = {
    "hybridgraphrag": CKPT / "hybridgraphrag-hotpotqa.jsonl",
    "naive_rag": CKPT / "naive_rag-hotpotqa.jsonl",
    "graphrag_local": CKPT / "graphrag_local-hotpotqa.jsonl",
    "graphrag_global": CKPT / "graphrag_global-hotpotqa.jsonl",
}


def load_checkpoint(path):
    """Load JSONL checkpoint into {id: {prediction, ground_truth}} dict."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                data[rec["id"]] = {
                    "prediction": rec.get("prediction", ""),
                    "ground_truth": rec.get("ground_truth", ""),
                }
    return data


def main():
    print("=" * 60 + "\nBERTScore Evaluation (from checkpoints)\n" + "=" * 60)

    # Load all systems
    system_data = {}
    for name, path in SYSTEMS.items():
        if not path.exists():
            print(f"  SKIP {name}: {path} not found")
            continue
        system_data[name] = load_checkpoint(path)
        print(f"  Loaded {name}: {len(system_data[name])} samples")

    # Compute matched IDs across all loaded systems
    id_sets = [set(d.keys()) for d in system_data.values()]
    common = sorted(set.intersection(*id_sets))
    print(f"  Matched IDs: {len(common)}")

    # Compute BERTScore for each system
    results, f1s = {}, {}
    for name, data in system_data.items():
        preds = [data[q]["prediction"] or "no answer" for q in common]
        golds = [data[q]["ground_truth"] or "no answer" for q in common]
        bs = compute_bertscore(preds, golds)
        results[name] = {
            k: bs[k]
            for k in ["precision", "recall", "f1", "f1_std",
                       "n", "model", "elapsed_s"]
        }
        f1s[name] = bs["f1_per_sample"]

    # Paired t-tests: HybridGraphRAG vs each baseline
    tests = []
    for bl in ["naive_rag", "graphrag_local", "graphrag_global"]:
        if bl not in f1s:
            continue
        tt = paired_ttest_bertscore(
            f1s["hybridgraphrag"], f1s[bl], f"vs_{bl}"
        )
        tests.append(tt)
        sig = "p<0.05 *" if tt["sig"] else "n.s."
        print(f"  vs {bl}: t={tt['t']}, p={tt['p']} ({sig})")

    # Save results
    output = {
        **results,
        "matched_ids": len(common),
        "model": results["hybridgraphrag"]["model"],
        "seed": 42,
        "paired_ttest": tests,
    }
    path = OUT / "bertscore-results.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n  Saved: {path}")
    for name in system_data:
        r = results[name]
        print(f"  {name:<20} F1={r['f1']:.4f} std={r['f1_std']:.4f}")


if __name__ == "__main__":
    main()
