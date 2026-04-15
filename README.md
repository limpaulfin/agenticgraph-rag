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
  booktitle = {Proceedings of the 14th International Conference on Frontiers of
               Intelligent Computing: Theory and Applications (FICTA 2026)},
  series    = {Smart Innovation, Systems and Technologies},
  publisher = {Springer},
  address   = {London, UK},
  year      = {2026},
  note      = {To appear}
}
```

## Lizenz

MIT
