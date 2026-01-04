from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.vector_store import FaissVectorStore
from src.search import RAGSearch

rag_search = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_search
    store = FaissVectorStore("faiss_store")
    store.load()
    rag_search = RAGSearch(store)
    yield

app = FastAPI(title="RAG LLM API", lifespan=lifespan)

class Query(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "RAG API running"}

@app.post("/ask")
def ask(q: Query):
    return {"answer": rag_search.search_and_summarize(q.question, top_k=3)}

from fastapi import UploadFile, File
import os
from src.data_loader import load_all_documents

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()

    if ext not in ["pdf", "txt", "csv"]:
        return {"error": "Unsupported file type"}

    folder = {
        "pdf": "pdf_files",
        "txt": "txt_files",
        "csv": "csv_files"
    }[ext]

    path = os.path.join("data", folder, file.filename)

    with open(path, "wb") as f:
        f.write(await file.read())

    docs = load_all_documents("data")
    rag_search.store.build_from_documents(docs)

    return {"message": f"{file.filename} uploaded and indexed"}
from fastapi.staticfiles import StaticFiles
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
