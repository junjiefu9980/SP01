import os
from pathlib import Path
import json
import glob
from typing import List, Dict, Tuple

import numpy as np
import faiss
import fitz     # PyMuPDF: Reading PDF
from sentence_transformers import SentenceTransformer


# file
ROOT_DIR = Path(__file__).resolve().parents[1]  # SP01
DOC_DIR = str(ROOT_DIR / "data" / "docs")
OUT_DIR = str(ROOT_DIR / "data" / "index")

# vector model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# chunk
CHUNK_SIZE = 900
CHUNK_OVERLAP = 180


def extract_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    # extract pdf text as pages, and return
    doc = fitz.open(pdf_path)       # PyMuPDF
    pages = []
    for i in range(len(doc)):
        txt = doc[i].get_text("text") or ""     # avoid None
        txt = txt.replace("\x00", "").strip()   # avoid blank and line break
        if txt:
            pages.append((i+1, txt))
    return pages


def read_text_file(path: str) -> str:
    # read txt/md file
    with open(path, "r", encoding = "utf-8", errors = "ignore") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Dict]:
    # cut the text to chunks with overlap
    chunks: List[Dict] = []
    start = 0
    n = len(text)

    # cover the whole text from 'start'
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append({"start": start, "end": end, "text": chunk})

        start = end - overlap
        if start < 0:
            start = 0
        if end == n:
            break

    return chunks


def l2_normalize(x: np.ndarray)  -> np.ndarray: # logarithm, for normalization
    # each raw norm; keep 2D shape
    norm = np.linalg.norm(x, axis = 1, keepdims=True) + 1e-12   # avoid 0, return (N,1)
    return x / norm


def main():
    # file
    os.makedirs(OUT_DIR, exist_ok = True)

    # 1. collect document files by recursion and only keep PDF, txt, docx
    files = sorted(glob.glob(os.path.join(DOC_DIR, "**/*.*"), recursive = True))
    files = [p for p in files if p.lower().endswith((".pdf", ".txt", ".docx", ".md"))]

    if not files:
        raise RuntimeError(f"No pdf/txt/docx files found in {DOC_DIR}")

    print(f"[INFO] Found {len(files)} documents")

    # 2. loading embedding model
    model = SentenceTransformer(MODEL_NAME)
    metas: List[Dict] = []
    texts: List[str] = []
    chunk_id = 0

    # 3. extract text to chunk and save to metas/texts
    for fp in files:
        ext = os.path.splitext(fp)[1].lower()
        fp_norm = fp.replace("\\", "/")

        if ext == ".pdf":
            pages = extract_pdf_pages(fp)

            # cut chunk for text in each page
            for page_no, page_text in pages:
                for c in chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP):
                    metas.append({
                        "chunk_id": chunk_id,
                        "doc_path": fp_norm,
                        "doc_type": "pdf",
                        "page": page_no,    # page number
                        "start": c["start"],    # strat point of this chunk in page-text
                        "end": c["end"],
                        "text": c["text"]    # text of this chunk
                    })
                    texts.append(c["text"])
                    chunk_id += 1
        else:
            raw = read_text_file(fp)    # for txt/md, read the whole file to a str

            for c in chunk_text(raw, CHUNK_SIZE, CHUNK_OVERLAP):
                metas.append({
                    "chunk_id": chunk_id,
                    "doc_path": fp_norm,
                    "doc_type": ext.replace(".", ""),    # .txt to txt
                    "page": None,    # no page for un-pdf file
                    "start": c["start"],
                    "end": c["end"],
                    "text": c["text"]
                })
    print(f"[INFO] Total chunks: {len(metas)}")
    print("[INFO] Embedding chunk ...")

    # 4. vector embedding
    embs = model.encode(texts, batch_size=64, show_progress_bar=True)   # (N, dim) array
    embs = np.asarray(embs, dtype = "float32")   # as FAISS requirements
    embs = l2_normalize(embs)     # for cosine retrieval by IndexFlatIP

    # 5. build and save FAISS retrival
    dim = embs.shape[1]     # vector dim
    index = faiss.IndexFlatIP(dim)      # IP=Inner Product
    index.add(embs)     # as order add all vectors to the retrieval
    faiss.write_index(index, os.path.join(OUT_DIR, "faiss.index"))      # FAISS lib function, write into binary file

    # 6. save meta.json for each row, easy to read as row
    meta_path = os.path.join(OUT_DIR, "meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")       # json.dumps: dict -> json str

    # 7. save manifest.json to record settings
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "num_chunks": len(metas),
            "doc_dir": DOC_DIR,
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved index files to: {OUT_DIR}")
    print("[OK] faiss.index / meta.jsonl / manifest.json")


if __name__ == "__main__":
    main()

