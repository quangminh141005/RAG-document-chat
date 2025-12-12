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
    
