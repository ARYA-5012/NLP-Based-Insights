"""
FastAPI Application — EarningsInsight AI Backend
Production-grade REST API for financial transcript analysis.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from src.api.routers import search, company, trends, compare, analyze


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("🚀 EarningsInsight AI starting up...")
    # Initialize vector store connection on startup
    try:
        from src.retrieval.vector_store import get_collection
        collection = get_collection()
        logger.info(f"Vector store ready. Documents: {collection.count()}")
    except Exception as e:
        logger.warning(f"Vector store not initialized: {e}")
    yield
    logger.info("EarningsInsight AI shutting down.")


app = FastAPI(
    title="EarningsInsight AI",
    description=(
        "Production NLP system that extracts actionable intelligence "
        "from SEC earnings call transcripts. Powered by semantic search, "
        "BERTopic, and GPT-based summarization."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS (allow frontend) ──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://earnings-insights.vercel.app",
        "*",  # Remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register Routers ───────────────────────────────────────────
app.include_router(search.router, prefix="/api", tags=["Search"])
app.include_router(company.router, prefix="/api", tags=["Company"])
app.include_router(trends.router, prefix="/api", tags=["Trends"])
app.include_router(compare.router, prefix="/api", tags=["Compare"])
app.include_router(analyze.router, prefix="/api", tags=["Analyze"])


# ─── Health Check ────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "EarningsInsight AI",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check with component status."""
    status = {"api": "healthy"}

    try:
        from src.retrieval.vector_store import get_collection_stats
        stats = get_collection_stats()
        status["vector_store"] = {"status": "connected", "documents": stats["count"]}
    except Exception as e:
        status["vector_store"] = {"status": "disconnected", "error": str(e)}

    status["openai"] = {
        "status": "configured" if os.getenv("OPENAI_API_KEY") else "not_configured"
    }

    return status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
