# AgenticGraph-RAG

## Zusammenfassung

AgenticGraph-RAG ist eine mehrstufige Pipeline fuer Frage-Antwort-Systeme. Das System kombiniert Wissensgraph-Konstruktion, Community-Erkennung und hybride Abruf-Strategien, um Multi-Hop-Fragen zu beantworten. Die Pipeline wird auf HotpotQA und MuSiQue evaluiert.

## Systemanforderungen

- Python 3.10+
- R 4.x (nur fuer Visualisierung)
- OpenAI API-Schluessel (fuer LLM-Aufrufe)
- Linux/macOS empfohlen

## Installation

```bash
# Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate

# Abhaengigkeiten installieren
pip install -r requirements.txt

# spaCy Modell herunterladen
python -m spacy download en_core_web_sm

# API-Schluessel setzen
export OPENAI_API_KEY="sk-..."
```

## Ausfuehrung

```bash
# Experiment mit 5 Beispielen (schneller Test)
python src/main.py --task experiment --n 5 --datasets hotpotqa

# Volles Experiment (n=1000, zwei Datensaetze)
python src/main.py --task experiment --n 1000 --datasets hotpotqa musique
```

### Visualisierung (R)

```bash
Rscript src/visualization/fig-main-results.R
Rscript src/visualization/fig-ablation.R
```

## Projektstruktur

```
agenticgraph-rag/
├── src/                      # Python Pipeline
│   ├── main.py               # Einstiegspunkt
│   ├── m00_logger.py         # JSONL Logger
│   ├── m01_data_loader.py    # Daten laden
│   ├── m02_*.py              # Baseline (Naive RAG, GraphRAG)
│   ├── m03_*.py              # Wissensgraph-Konstruktion
│   ├── m05_*.py              # Community-Erkennung (Leiden)
│   ├── m06_*.py              # Community-Zusammenfassung
│   ├── m07_*.py              # Hybride Abruf-Strategie
│   ├── m08_*.py              # Antwort-Generierung
│   ├── m09_*.py              # Experiment-Runner
│   ├── m10_*.py              # Ablation + Statistik
│   ├── m11_*.py              # BERTScore Evaluierung
│   ├── prompts/              # LLM Prompt-Vorlagen
│   └── visualization/        # R Skripte fuer Abbildungen
├── data/
│   ├── hotpotqa/             # HotpotQA Datensatz (Sample)
│   └── musique/              # MuSiQue Datensatz (Sample)
├── requirements.txt
└── README.md
```

## Datensatz

- **HotpotQA**: Multi-Hop QA Datensatz (Yang et al., 2018). Sample: 1000 Fragen.
- **MuSiQue**: Multi-Step QA Datensatz (Trivedi et al., 2022). Sample: 1000 Fragen.

Die vollstaendigen Datensaetze koennen von den Originalquellen heruntergeladen werden:
- HotpotQA: https://hotpotqa.github.io/
- MuSiQue: https://github.com/StonyBrookNLP/musique

## Zitierung

```bibtex
@inproceedings{nguyen2026agenticgraphrag,
  title     = {AgenticGraph-RAG: A Multi-Layer Agentic Framework
               for Adaptive Retrieval-Augmented Generation
               over Knowledge Graphs},
  author    = {Nguyen, Quyen Pham Thi Thanh
               and Lim, Paul Fong Jia Pheng
               and Nguyen, Binh Thanh},
  booktitle = {Proceedings of HybridAIMS Workshop
               at CAiSE 2026},
  year      = {2026},
  note      = {Paper 468}
}
```

## Lizenz

MIT License. Siehe [LICENSE](LICENSE).
