"""Statistical tests: McNemar (EM) + paired t-test (F1).
Ablation: full HybridGraphRAG vs each variant (matched IDs).
"""
import json, sys
import numpy as np
from scipy import stats
from pathlib import Path

CKPT = Path(__file__).parent.parent / 'output' / 'checkpoints'
OUT = Path(__file__).parent.parent / 'output'
sys.path.insert(0, str(Path(__file__).parent))
from m02_evaluate import exact_match, f1_score

ABLATION = {'no_kg': 'ablation-no_kg-hotpotqa.jsonl',
            'no_community': 'ablation-no_community-hotpotqa.jsonl',
            'no_vector': 'ablation-no_vector-hotpotqa.jsonl',
            'no_fusion': 'ablation-no_fusion-hotpotqa.jsonl'}


def _load(path):
    r = {}
    for line in open(path):
        if line.strip():
            d = json.loads(line); r[d['id']] = d
    return r


def mcnemar(fp, vp, label):
    full, var = _load(fp), _load(vp)
    ids = sorted(set(full) & set(var))
    b = c = 0
    for q in ids:
        fe = exact_match(full[q]['prediction'], full[q]['ground_truth'])
        ve = exact_match(var[q]['prediction'], var[q]['ground_truth'])
        if fe == 1.0 and ve == 0.0: b += 1
        elif fe == 0.0 and ve == 1.0: c += 1
    chi2 = (abs(b - c) - 1)**2 / (b + c) if b + c > 0 else 0.0
    p = 1 - stats.chi2.cdf(chi2, df=1) if b + c > 0 else 1.0
    return {'vs': label, 'n': len(ids), 'b': int(b), 'c': int(c),
            'chi2': round(float(chi2), 4), 'p': round(float(p), 6),
            'sig': bool(p < 0.05)}


def paired_ttest(fp, vp, label):
    full, var = _load(fp), _load(vp)
    ids = sorted(set(full) & set(var))
    ff = [f1_score(full[i]['prediction'], full[i]['ground_truth']) for i in ids]
    vf = [f1_score(var[i]['prediction'], var[i]['ground_truth']) for i in ids]
    t, p = stats.ttest_rel(ff, vf)
    d = np.array(ff) - np.array(vf)
    ci = stats.t.interval(0.95, len(d) - 1, loc=np.mean(d), scale=stats.sem(d))
    return {'vs': label, 'n': len(ids), 't': round(float(t), 4),
            'p': round(float(p), 6), 'mean_diff': round(float(np.mean(d)), 4),
            'ci95': [round(float(ci[0]), 4), round(float(ci[1]), 4)],
            'sig': bool(p < 0.05)}


def run_all():
    ref = str(CKPT / 'hybridgraphrag-hotpotqa.jsonl')
    results = {'mcnemar': [], 'paired_ttest': []}
    print('\n--- Ablation Tests (HotpotQA, n=200 matched IDs) ---')
    for name, fname in ABLATION.items():
        vp = str(CKPT / fname)
        mc = mcnemar(ref, vp, name)
        tt = paired_ttest(ref, vp, name)
        results['mcnemar'].append(mc)
        results['paired_ttest'].append(tt)
        sig = lambda r: 'p<0.05' if r['sig'] else 'n.s.'
        print(f"  {name}: McNemar p={mc['p']}({sig(mc)}) "
              f"t-test p={tt['p']}({sig(tt)}) CI={tt['ci95']}")
    with open(OUT / 'statistical_tests.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved statistical_tests.json')
    return results


if __name__ == '__main__':
    print('Running statistical tests...'); run_all(); print('Done.')
