"""Publication figures for HybridGraph-RAG. CUD palette. Data-driven."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

CUD = {'blue': '#0072B2', 'orange': '#E69F00', 'sky': '#56B4E9',
       'black': '#000000', 'vermilion': '#D55E00'}
BASE = Path(__file__).parent.parent
FIG = BASE.parent / 'LaTeX' / 'figures'
BL_KEYS = ['naive_rag', 'graphrag_local', 'graphrag_global']
plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

def _save(fig, n):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / n); plt.close(fig); print(f'  Saved {n}')

def _jl(n):
    with open(BASE / 'output' / n) as f: return json.load(f)

def _met():
    bl = _jl('baselines-all.json')['experiments']['hotpotqa']
    hgr = _jl('ablation.json')['fair_comparison']['full_model_n200']
    r = {k: bl[k]['metrics'] for k in BL_KEYS}
    r['hgr'] = hgr
    return r

def _clean(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

def _lbl(ax, bars, vals, fs=8):
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2., b.get_height()+.004,
                f'{v:.3f}', ha='center', va='bottom', fontsize=fs, fontweight='bold')

def fig_main():
    m = _met(); ks = [*BL_KEYS, 'hgr']
    v = [m[k]['recall_at_k'] for k in ks]
    c = [CUD['sky'], CUD['blue'], CUD['blue'], CUD['orange']]
    lb = ['NaiveRAG', 'GraphRAG\n(Local)', 'GraphRAG\n(Global)', 'HybridGraph\n-RAG']
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    _lbl(ax, ax.bar(lb, v, color=c, edgecolor=CUD['black'], lw=.8, width=.6), v)
    ax.set_ylabel('Recall@5'); ax.set_ylim(.65, .84); _clean(ax)
    ax.set_title('Retrieval Quality (HotpotQA, n=200)'); _save(fig, 'fig-main-results.pdf')

def fig_ablation():
    d = _jl('ablation.json'); full = d['fair_comparison']['full_model_n200']
    order = ['no_kg', 'no_community', 'no_vector', 'no_fusion']
    lb = ['Full', 'w/o\nKG', 'w/o\nComm', 'w/o\nVec', 'w/o\nFus']
    sig = {}
    sp = BASE / 'output' / 'statistical_tests.json'
    if sp.exists():
        for t in json.loads(sp.read_text()).get('paired_ttest', []):
            sig[t['vs']] = t['sig']
    panels = [('Exact Match', 'em', .38, .60),
              ('F1 Score', 'f1', .52, .73), ('Recall@5', 'recall_at_k', .68, .84)]
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.2))
    for ax, (title, key, ylo, yhi) in zip(axes, panels):
        vals = [full[key]] + [d['variants'][v]['metrics'][key] for v in order]
        bars = ax.bar(lb, vals, color=[CUD['orange']]+[CUD['sky']]*4,
                       edgecolor=CUD['black'], lw=.6, width=.65)
        _lbl(ax, bars, vals, fs=7)
        for i, v in enumerate(order):
            if sig.get(v):
                ax.text(i+1, vals[i+1]+.018, '*', ha='center', fontsize=12,
                        color=CUD['vermilion'])
        ax.set_title(title, fontsize=10); ax.set_ylim(ylo, yhi); _clean(ax)
    fig.suptitle('Ablation Study (HotpotQA, n=200)', fontsize=11, y=1.02)
    plt.tight_layout(); _save(fig, 'fig-ablation.pdf')

def fig_comparison():
    m = _met(); ks = [*BL_KEYS, 'hgr']
    em = [m[k]['em'] for k in ks]; f1 = [m[k]['f1'] for k in ks]
    r5 = [m[k]['recall_at_k'] for k in ks]
    lb = ['NaiveRAG', 'GraphRAG-L', 'GraphRAG-G', 'HybridGraph-RAG']
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x, w = np.arange(4), .22
    ax.bar(x-w, em, w, label='EM', color=CUD['sky'], edgecolor=CUD['black'], lw=.5)
    ax.bar(x, f1, w, label='F1', color=CUD['blue'], edgecolor=CUD['black'], lw=.5)
    ax.bar(x+w, r5, w, label='R@5', color=CUD['orange'], edgecolor=CUD['black'], lw=.5)
    ax.set_ylabel('Score'); ax.set_xticks(x); ax.set_xticklabels(lb, fontsize=8)
    ax.legend(frameon=False); ax.set_ylim(.4, .9); _clean(ax)
    ax.set_title('Multi-Metric Comparison (HotpotQA, n=200)')
    _save(fig, 'fig-comparison.pdf')

if __name__ == '__main__':
    print('Generating figures...'); fig_main(); fig_ablation(); fig_comparison()
    print('Done.')
