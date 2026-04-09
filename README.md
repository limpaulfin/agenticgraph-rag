# AgenticGraph-RAG

Multi-Hop QA Pipeline mit Graph-Augmented Retrieval und Adversarial Verification.

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
export OPENAI_API_KEY="sk-..."
```

## Ausfuehrung

```bash
python src/main.py --task experiment --n 5 --datasets hotpotqa
```

## Zitierung

```bibtex
@inproceedings{lam2026agenticgraphrag,
  title     = {AgenticGraph-RAG: Multi-Hop QA via Graph-Augmented
               Retrieval and Adversarial Tool-Augmented Generation},
  author    = {Lam, Thanh-Phong and Tran, Viet-Tam and Nguyen, Huynh-Anh-Vu},
  booktitle = {HybridAIMS Workshop at CAiSE 2026},
  year      = {2026}
}
```

## Lizenz

MIT
