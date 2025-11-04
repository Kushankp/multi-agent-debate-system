import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# load env for any route-level config (Agent_A already loads .env too)
load_dotenv()

# Import the agent that handles embeddings + Pinecone
from app.services.Agent_A import search as agent_search, health as agent_health

# ---------- FastAPI app ----------
app = FastAPI(title="RAG Retrieval API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request/Response models ----------
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    fetch_k: Optional[int] = None
    namespace: Optional[str] = None

class ChunkResult(BaseModel):
    id: str
    page: str
    chunk_index: int
    score: float
    snippet: str
    full_text: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[ChunkResult]

# ---------- Search endpoint ----------
@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    try:
        out = agent_search(
            query=req.query,
            top_k=req.top_k,
            fetch_k=req.fetch_k,
            namespace=req.namespace,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    # Validate/normalize to match Pydantic response model
    return SearchResponse(query=out.get("query", ""), results=[ChunkResult(**r) for r in out.get("results", [])])

# ---------- Health ----------
@app.get("/health")
def health():
    return agent_health()
