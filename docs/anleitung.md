# Anleitung

## Schnellstart

1. Repository klonen
2. Virtuelle Umgebung erstellen: `python -m venv venv && source venv/bin/activate`
3. Abhaengigkeiten: `pip install -r requirements.txt`
4. spaCy Modell: `python -m spacy download en_core_web_sm`
5. API-Schluessel: `export OPENAI_API_KEY="sk-..."`
6. Test: `python src/main.py --task experiment --n 5 --datasets hotpotqa`

## Pipeline-Module

| Modul | Beschreibung |
|-------|-------------|
| m00 | JSONL Logger |
| m01 | Daten laden und normalisieren |
| m02 | Baseline: Naive RAG + GraphRAG |
| m03 | Wissensgraph bauen (spaCy NER + Dependency) |
| m05 | Community-Erkennung (Leiden-Algorithmus) |
| m06 | Community-Zusammenfassung (LLM) |
| m07 | Hybride Abruf-Strategie (lokal + global + Passage) |
| m08 | Antwort-Generierung (CoT + XML) |
| m09 | Experiment-Runner (alle Methoden vergleichen) |
| m10 | Ablation + statistische Tests |
| m11 | BERTScore Evaluierung |

## Datenformat

Eingabe: JSONL Dateien mit Feldern `question`, `answer`, `context`.

Ausgabe: JSON Dateien in `output/` mit Metriken (EM, F1, BERTScore).

## Umgebungsvariablen

| Variable | Beschreibung |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API-Schluessel (erforderlich) |

## Visualisierung

R 4.x mit ggplot2. Skripte in `src/visualization/`.

```bash
Rscript src/visualization/fig-main-results.R
```
