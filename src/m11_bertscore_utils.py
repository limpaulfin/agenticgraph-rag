"""BERTScore utilities: compute scores and paired t-test."""

import numpy as np
from scipy import stats

BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
BERTSCORE_FALLBACK = "distilbert-base-uncased"


def compute_bertscore(predictions, gold_answers, model_type=BERTSCORE_MODEL):
    """Compute BERTScore (P, R, F1) with fallback model."""
    import time
    from bert_score import score

    print(f"  Computing BERTScore ({model_type}, n={len(predictions)})...")
    t0 = time.time()
    try:
        P, R, F1 = score(predictions, gold_answers, lang="en",
                         model_type=model_type, verbose=True,
                         rescale_with_baseline=True)
    except Exception as e:
        print(f"  ERROR: {e}. Falling back to {BERTSCORE_FALLBACK}...")
        model_type = BERTSCORE_FALLBACK
        P, R, F1 = score(predictions, gold_answers, lang="en",
                         model_type=model_type, verbose=True)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return {
        "precision": float(P.mean()), "recall": float(R.mean()),
        "f1": float(F1.mean()), "f1_std": float(F1.std()),
        "f1_per_sample": [float(x) for x in F1.tolist()],
        "n": len(predictions), "model": model_type,
        "elapsed_s": round(elapsed, 2),
    }


def paired_ttest_bertscore(f1_a, f1_b, label):
    """Paired t-test on per-sample BERTScore F1 arrays."""
    a, b = np.array(f1_a), np.array(f1_b)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    ci = stats.t.interval(0.95, len(d) - 1, loc=np.mean(d), scale=stats.sem(d))
    return {"vs": label, "n": len(a), "t": round(float(t), 4),
            "p": round(float(p), 6), "mean_diff": round(float(np.mean(d)), 4),
            "ci95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
            "sig": bool(p < 0.05)}
