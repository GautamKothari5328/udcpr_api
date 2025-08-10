from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import os
from transformers import pipeline

# --------------------------
# Load FAISS index & metadata
# --------------------------
INDEX_PATH = "./udcpr_index.faiss"
META_PATH = "./chunks_metadata.pkl"

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

# Load metadata (chunk text, page numbers)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

# Load embedding model (LOCAL)
embed_model = SentenceTransformer("./all-MiniLM-L6-v2")  # make sure it's downloaded

# Load summarization LLM (FREE model from HuggingFace)
summarizer = pipeline(
    "summarization",
    model="./models/distilbart",  # Use locally downloaded model
    tokenizer="./models/distilbart"
)

# --------------------------
# FastAPI app
# --------------------------
app = FastAPI(title="UDCPR RAG API", version="2.0")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

# --------------------------
# Helper Functions
# --------------------------
def clean_text(txt: str) -> str:
    """Remove tabs, newlines, and excessive spaces."""
    return " ".join(txt.replace("\t", " ").replace("\n", " ").split())

def retrieve_chunks(question, top_k):
    """Retrieve top_k relevant chunks for the given question."""
    query_emb = embed_model.encode([question])
    distances, indices = index.search(np.array(query_emb).astype("float32"), top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        results.append({
            "score": float(score),
            "page": metadata[idx]["page"],
            "text": clean_text(metadata[idx]["text"])
        })
    return results

def refine_answer(question, context_text):
    """Use the summarization model to generate a refined answer."""
    prompt = (
        f"You are an assistant for architects. "
        f"Question: {question}\n\n"
        f"Here is the extracted regulation text from the UDCPR manual:\n"
        f"{context_text}\n\n"
        f"Provide a clear, concise, human-readable answer with bullet points for conditions. "
        f"Do not include raw references or regulation numbers unless absolutely necessary."
    )

    summary = summarizer(prompt, max_length=400, min_length=80, do_sample=False)[0]["generated_text"]
    return clean_text(summary)

# --------------------------
# API Endpoint
# --------------------------
@app.post("/ask")
def ask_question(request: QueryRequest):
    # Step 1: Retrieve matching chunks
    chunks = retrieve_chunks(request.question, request.top_k)

    # Step 2: Combine text from chunks for refinement
    combined_text = " ".join([chunk["text"] for chunk in chunks])

    # Step 3: Generate refined answer
    refined = refine_answer(request.question, combined_text)

    return {
        "question": request.question,
        "refined_answer": refined,
        "sources": chunks  # Optional: keep for transparency
    }
