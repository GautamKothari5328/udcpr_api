# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load FAISS index and metadata at startup
INDEX_PATH = "udcpr_index.faiss"        # path to saved index
CHUNKS_PATH = "chunks_metadata.pkl"     # path to saved chunks

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, 'rb') as f:
    chunks = pickle.load(f)

print("Loading embedding model...")
embed_model = SentenceTransformer('./all-MiniLM-L6-v2')

# FastAPI app
app = FastAPI(title="UDCPR Architect Q&A API")

class QuestionRequest(BaseModel):
    question: str
    k: int = 3

@app.post("/ask")
def ask_question(req: QuestionRequest):
    q_vec = embed_model.encode([req.question]).astype('float32')
    D, I = index.search(q_vec, req.k)

    results = []
    for score, idx in zip(D[0], I[0]):
        item = chunks[idx]
        results.append({
            "score": float(score),
            "page": item['page'],
            "text": item['text']
        })
    
    return {"question": req.question, "answers": results}
