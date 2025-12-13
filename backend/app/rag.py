import os 
import uuid
from typing import List, Dict, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from pypdf import PdfReader
from openai import OpenAI

from .settings import settings # from settings.py

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _load_text_from_pdf(file_path: str) -> str: # internal use
    reader = PdfReader(file_path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)

# Load text
def _load_text(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        return _load_text_from_pdf(file_path)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# Spliting text into meaningful(?) chunks  
def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = " ".join(text.split())
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += chunk_size - overlap # retrive some words
    return [c for c in chunks if c.strip()]

# text -> vector
def _embed(texts: List[str]) -> List[List(float)]:
    """
    Send text to OpenAI embedding model and return the numerical vectors, return object contain data: .embedding and .index
    """
    resp = client.embeddings.create(
        model=settings.OPENAI_EMBED_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data] # select .embedding, ignore .index in the JSON format

def get_collection():
    _ensure_dir(settings.CHROMA_DIR)
    ch = chromadb.PersistentClient(
        path=settings.CHROMA_DIR,
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    return ch.get_or_create_collection(settings.COLLECTION)

def ingest_file(file_path: str, original_name: str) -> Dict:
    text = _load_text(file_path)
    chunks = _chunk_text(text)
    vectors = _embed(chunks)

    col = get_collection()
    doc_id = str(uuid.uuid4())

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_id": doc_id, "source": original_name, "chunk": i} 
        for i in range(len(chunks))
    ]
    col.add(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas) # store everything in chromaDB
    return {"doc_id": doc_id, "chunks": len(chunks), "source": original_name}

# retrieve the relevant part base on the question
def retrieve(question: str, top_k: int) -> Tuple[List[str], List[Dict]]:
    """
    Take the question, get 2 seperate lists include metadata and distances, filter it to get a clean list with docs (source, chunk and score)
    """
    q_emb = _embed([question])[0]
    col = get_collection()
    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        includes=["documents", "metadatas", "distances"]
    )
    docs = res["documents"][0] if res.get("documents") else []
    metas = res["metadatas"][0] if res.get("metadatas") else []
    dists = res["distances"][0] if res.get("distances") else []

    sources = []
    for m, d in zip(metas, dists):
        sources.append({
            "source": m.get("source"),
            "chunk": m.get("chunk"),
            "score": float(d)
        })
    return docs, sources

def answer(question: str) -> Dict:
    contexts, sources = retrieve(question, settings.TOP_K)
    context_blocks = "\n\n---\n\n".join(contexts) if contexts else "(no context found)"

    system = (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "If the context doesn't contain the answer, say you don't know. "
        "Be concise and include bullet points when helpful."
    )

    user = f"CONTEXT:\n{context_blocks}\n\nQUESTION:\n{question}"

    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.2,
    )

    return {
        "answer": resp.choices[0].message.content.strip(),
        "sources": sources
    }
