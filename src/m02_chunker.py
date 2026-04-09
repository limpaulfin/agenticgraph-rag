"""Text chunking and passage extraction for RAG pipelines."""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
MODEL_NAME = "all-MiniLM-L6-v2"


def extract_passages(record: dict) -> list[str]:
    """Extract text passages from HotpotQA/MuSiQue context field."""
    passages = []
    for title_sents in record.get("context", []):
        if isinstance(title_sents, list) and len(title_sents) >= 2:
            title, sents = title_sents[0], title_sents[1]
            text = f"{title}: {' '.join(sents)}" if isinstance(sents, list) else f"{title}: {sents}"
            passages.append(text)
    return passages


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def build_chunks(record: dict) -> list[str]:
    """Build all chunks for a single question's context."""
    result = []
    for passage in extract_passages(record):
        result.extend(chunk_text(passage))
    return result


def retrieve_top_k(model: SentenceTransformer, question: str, chunks: list[str], k: int = TOP_K) -> list[str]:
    """Encode chunks + question, return top-k nearest chunks via FAISS."""
    if not chunks:
        return []
    chunk_emb = model.encode(chunks, show_progress_bar=False).astype(np.float32)
    q_emb = model.encode([question], show_progress_bar=False).astype(np.float32)
    index = faiss.IndexFlatL2(chunk_emb.shape[1])
    index.add(chunk_emb)
    _, indices = index.search(q_emb, min(k, len(chunks)))
    return [chunks[i] for i in indices[0]]
