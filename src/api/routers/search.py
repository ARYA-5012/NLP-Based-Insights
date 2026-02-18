from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from src.retrieval.vector_store import query_collection
from src.synthesis.summarizer import generate_answer

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    n_results: int = 5
    source_filter: Optional[str] = None  # 'clothing_reviews' or 'twitter_support'
    category_filter: Optional[str] = None # For reviews
    issue_type_filter: Optional[str] = None # For tweets

class SearchResult(BaseModel):
    text: str
    metadata: dict
    score: float

class QA_Request(BaseModel):
    question: str
    source_filter: Optional[str] = None

class QA_Response(BaseModel):
    answer: str
    sources: List[str]

@router.post("/", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """Semantic search over reviews and tweets."""
    where_filter = {}
    if request.source_filter:
        where_filter["source"] = request.source_filter
    if request.category_filter:
        where_filter["category"] = request.category_filter
    if request.issue_type_filter:
        where_filter["issue_type"] = request.issue_type_filter
        
    # If no filters, pass None to query_collection
    if not where_filter:
        where_filter = None

    results = query_collection(request.query, n_results=request.n_results, where=where_filter)
    
    response = []
    for i in range(len(results["documents"][0])):
        response.append(SearchResult(
            text=results["documents"][0][i],
            metadata=results["metadatas"][0][i],
            score=results["distances"][0][i]  # Distance, lower is better
        ))
    return response

@router.post("/ask", response_model=QA_Response)
async def ask_question(request: QA_Request):
    """RAG-powered Q&A on the customer voice data."""
    context_filter = {}
    if request.source_filter:
        context_filter["source"] = request.source_filter
        
    if not context_filter:
        context_filter = None
        
    # 1. Retrieve
    results = query_collection(request.question, n_results=5, where=context_filter)
    context_texts = results["documents"][0]
    
    # 2. Sythesize
    # Adjust prompt based on source? For now generic RAG is fine.
    answer = generate_answer(request.question, context_texts)
    
    return QA_Response(answer=answer, sources=context_texts)
