# HotpotQA Dataset

## Tổng quan

HotpotQA là benchmark multi-hop question answering yêu cầu reasoning qua nhiều đoạn văn Wikipedia.
Bài báo gốc: Yang et al. (2018) - "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering".

- **Nguồn**: http://curtis.ml.cmu.edu/datasets/hotpot/
- **Setting**: Distractor (10 gold + distractors per question)
- **Split**: Dev set (hard questions only)
- **License**: CC-BY-SA-4.0

## Thống kê

| Metric | Giá trị |
|--------|---------|
| Tổng số câu hỏi (dev) | 7,405 |
| Bridge questions | 5,918 (79.9%) |
| Comparison questions | 1,487 (20.1%) |
| Difficulty level | All hard |
| Sample subset | 1,000 (800 bridge + 200 comparison) |

## Cấu trúc file

```
data/hotpotqa/
  hotpot_dev_distractor_v1.json   # Raw JSON từ CMU (45MB)
  hotpotqa-dev-full.jsonl          # Full dev set, JSONL format (7,405 lines)
  hotpotqa-dev-sample-1000.jsonl   # Stratified sample 1,000 questions
  README.md                        # File này
```

## Format JSONL

Mỗi dòng là một JSON object:

```json
{
  "id": "5a8b57f25542995d1e6f1371",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "answer": "yes",
  "type": "comparison",
  "level": "hard",
  "supporting_facts": [["Scott Derrickson", 0], ["Ed Wood", 0]],
  "context": [["title1", ["sent1", "sent2", ...]], ...]
}
```

### Các trường

| Trường | Mô tả |
|--------|-------|
| id | ID duy nhất (hex string) |
| question | Câu hỏi multi-hop |
| answer | Đáp án ngắn (span) |
| type | `bridge` hoặc `comparison` |
| level | Độ khó (`hard` cho dev set) |
| supporting_facts | Danh sách [title, sentence_idx] cần để trả lời |
| context | 10 đoạn văn Wikipedia [[title, [sentences]], ...] |

## Loại câu hỏi

- **Bridge**: Yêu cầu thực thể trung gian để kết nối 2 hop reasoning. Ví dụ: "What year did the sequel to [Book X] come out?" (cần biết Book X -> tìm sequel -> tìm năm).
- **Comparison**: So sánh thuộc tính giữa 2 thực thể. Ví dụ: "Which airport is owned by a higher government?" (cần tra cứu cả 2 sân bay).

## Sampling strategy

- Seed: 42 (reproducible)
- Stratified: giữ nguyên tỷ lệ bridge/comparison (~80/20)
- Sample 1,000 từ 7,405 cho development/quick evaluation
- Full set cho final evaluation

## Sử dụng trong pipeline

```python
import json

# Đọc sample
questions = []
with open('hotpotqa-dev-sample-1000.jsonl') as f:
    for line in f:
        questions.append(json.loads(line))

# Lọc theo loại
bridge_qs = [q for q in questions if q['type'] == 'bridge']
comparison_qs = [q for q in questions if q['type'] == 'comparison']
```

## Tham khảo

Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., & Manning, C. D. (2018).
HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.
*Proceedings of EMNLP 2018*. https://doi.org/10.18653/v1/D18-1259
