import os
import json
from pathlib import Path

import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from sympy import continued_fraction_reduce

# file
ROOT_DIR = Path(__file__).resolve().parents[1]   # SP01
INDEX_DIR = ROOT_DIR / "data" / "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def l2_normalize(x):
    # normalization the vector, to cosine
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm

def load_meta(path):
    # read meta.jsonl, order = faiss.index
    metas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

# loading index, metas and embedding model
index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
metas = load_meta(INDEX_DIR / "meta.jsonl")
model = SentenceTransformer(MODEL_NAME)

def format_sources(hits):
    # turn hits to sources text
    lines = []
    for rank, score, m in hits:
        snippet = m["text"].replace("\n", " ")
        snippet = snippet[:240] + ("..." if len(snippet) > 240 else "")
        page = m.get("page")
        page_str = f"p.{page}" if page else ""
        lines.append(
            f"[S{rank} score={score:.4f} | chunk_id={m['chunk_id']} | {m['doc_path']} {page_str}\n"
            f"      {snippet}"
        )

def answer_from_evidence(hits):
    # mini answer to list the snippet of evidence, containing number
    bullets = []
    for rank, score, m in hits:
        page = m.get("page")
        page_str = f"p.{page}" if page else ""
        snippet = m["text"][:380].strip().replace("\n", " ")
        bullets.append(f"- [S{rank}] {page_str} {snippet}")

    return "As retrival snippet in material, \n\n" + "\n".join(bullets)

def rag(question:str, top_k: int):
    # input question, top_k returns the number of chunks
    # output answer text for mini answer, and the source text with pages and path
    question = (question or "").strip()
    if not question:
        return "", ""

    # 1) embedding, encode input with List[str], use []
    q_emb = model.encode([question])
    q_emb = np.asarray(q_emb, dtype="float32")
    q_emb = l2_normalize(q_emb)

    # 2) FAISS
    scores, idxs = index.search(q_emb, top_k)
    scores = scores[0].tolist()
    idxs = idxs[0].tolist()

    hits = []
    rank = 1
    for score, idx in zip(scores, idxs):
        if idx == -1:
            continue
        hits.append((rank, float(score), metas[idx]))
        rank += 1

    return answer_from_evidence(hits), format_sources(hits)

# ==== Gradio UI ====
with gr.Blocks(title="SP01 Minimal PDF RAG") as demo:
    gr.Markdown("# SP01 - Minimal RAG (PDF/Text to Q&A with Sources")

    q = gr.Textbox(label="Question", placeholder="Ask about your PDF manual...", lines=2)
    topk = gr.Slider(1, 10, value=5, step=1, label="Top K")
    btn = gr.Button("Ask")

    ans = gr.Textbox(label="Answer", lines=12)
    src = gr.Textbox(label="Source (chunk_id / page / snippet)", lines=14)

    btn.click(fn=rag, inputs=[q, topk], outputs=[ans, src])


if __name__ == "__main__":
    demo.launch()



