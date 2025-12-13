import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from .rag import ingest_file, answer
from .schemas import ChatRequest, ChatResponse

app = FastAPI(title="RAG Doc Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # request from any website
    allow_credentials=True, # cookies, tokens, authentication headers
    allow_methods=["*"], #
    allow_headers=["*"],
)

UPLOAD_DIR = "app/storage/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok"}
            
@app.post("/upload")
def upload(file: UploadFile = File(...)):
    # save upload file
    dst = os.path.join(UPLOAD_DIR, file.filename)
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)

    info = ingest_file(dst, file.filename)
    return {"message": "ingested", **info}

