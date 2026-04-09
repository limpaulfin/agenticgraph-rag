# MuSiQue Dataset

## Tổng quan

MuSiQue (Multihop Questions via Single-hop Question Composition) là bộ dữ liệu multi-hop QA
với 2-4 hops, thiết kế để đánh giá khả năng reasoning phức tạp trên knowledge graphs.

- **Nguồn**: Trivedi et al. (2022), HuggingFace: dgslibisey/MuSiQue
- **License**: CC-BY-4.0
- **Phiên bản**: musique_ans_v1.0 (answerable subset)
- **Split**: dev (validation)

## Thống kê

| Metric | Giá trị |
|--------|---------|
| Tổng records | 2,417 |
| 2-hop | 1,252 (51.8%) |
| 3-hop | 760 (31.4%) |
| 4-hop | 405 (16.8%) |
| Answerable | 2,417 (100%) |
| Paragraphs/record | 20 |

## Files

| File | Mô tả | Size |
|------|--------|------|
| `musique_ans_v1.0_dev.jsonl` | Raw data từ HuggingFace | 29MB |
| `musique-dev-full.jsonl` | Unified format (2,417 records) | ~30MB |
| `musique-dev-sample-1000.jsonl` | Stratified sample (seed=42) | ~12MB |

## Sample distribution (1000)

| Hop | Count | Ratio |
|-----|-------|-------|
| 2-hop | 518 | 51.8% |
| 3-hop | 314 | 31.4% |
| 4-hop | 168 | 16.8% |

## Unified Schema

```json
{
  "id": "2hop__460946_294723",
  "question": "Who is the spouse of the Green performer?",
  "answer": "Miquette Giraudy",
  "answer_aliases": [],
  "type": "2hop",
  "n_hops": 2,
  "supporting_facts": [["title", idx], ...],
  "context": [["title", ["paragraph_text"]], ...],
  "question_decomposition": [{"id": ..., "question": ..., "answer": ...}],
  "answerable": true,
  "dataset": "musique"
}
```

## So sánh với HotpotQA

| Feature | HotpotQA | MuSiQue |
|---------|----------|---------|
| Hops | 2 (bridge/comparison) | 2-4 (DAG structure) |
| Records (dev) | 7,405 | 2,417 |
| Reasoning | Bridge + Comparison | Compositional multi-hop |
| Difficulty | Hard | Harder (connected DAGs) |

## Tại sao chọn MuSiQue

Theo Perplexity DSS (2026-02-16): MuSiQue phù hợp hơn cho GraphRAG hierarchical community
summarization vì enforced connected DAGs và indispensable hops, tránh disconnected reasoning.
Được dùng trong ChainRAG (2025) và TCR-QF (2025) cho multi-hop RAG evaluation.

## Citation

Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2022).
MuSiQue: Multihop Questions via Single Hop Question Composition.
Transactions of the Association for Computational Linguistics, 10, 539-554.
