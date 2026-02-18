"""
Vector Store abstraction over ChromaDB (local) and Pinecone (production).
Handles collection management, document ingestion, and query execution.
"""
import os
from typing import List, Dict, Optional
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

_client = None
_collection = None


def get_collection(collection_name: str = "earnings_transcripts",
                   persist_dir: str = None):
    """Get or create a ChromaDB collection (singleton)."""
    global _client, _collection
    
    if _collection is not None:
        return _collection
    
    import chromadb
    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/embeddings")
    
    logger.info(f"Initializing ChromaDB at: {persist_dir}")
    _client = chromadb.PersistentClient(path=persist_dir)
    _collection = _client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"Collection '{collection_name}' ready. Count: {_collection.count()}")
    return _collection


def ingest_chunks(chunks: List[Dict], collection_name: str = "earnings_transcripts") -> int:
    """
    Ingest embedded chunks into ChromaDB.

    Args:
        chunks: List of dicts with "text", "embedding", "metadata".

    Returns:
        Number of chunks ingested.
    """
    collection = get_collection(collection_name)

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        # Build a unique ID: ticker_quarter_year_chunkN
        meta = chunk["metadata"]
        ticker = meta.get("ticker", "UNK")
        quarter = meta.get("quarter", "Q0")
        year = meta.get("year", 0)
        section = meta.get("section", "unknown")
        chunk_id = f"{ticker}_{quarter}_{year}_{section}_{i}"

        ids.append(chunk_id)
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])

        # ChromaDB metadata must be str/int/float/bool only
        clean_meta = {
            "ticker": str(meta.get("ticker", "")),
            "quarter": str(meta.get("quarter", "")),
            "year": int(meta.get("year", 0)),
            "section": str(meta.get("section", "")),
            "speaker": str(meta.get("speaker", "")),
            "role": str(meta.get("role", "")),
            "token_count": int(chunk.get("token_count", 0)),
        }
        metadatas.append(clean_meta)

    # ChromaDB upsert (add or update)
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
    
    logger.info(f"Ingested {len(ids)} chunks into '{collection_name}'. Total count: {collection.count()}")
    return len(ids)


def query_vectors(query_embedding: List[float], top_k: int = 10,
                  filters: Dict = None,
                  collection_name: str = "earnings_transcripts") -> List[Dict]:
    """
    Query ChromaDB for similar chunks.

    Args:
        query_embedding: Query vector.
        top_k: Number of results to return.
        filters: ChromaDB where clause, e.g. {"ticker": "AAPL"}.

    Returns:
        List of results: [{"id", "text", "metadata", "distance", "score"}, ...]
    """
    collection = get_collection(collection_name)
    
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if filters:
        query_params["where"] = filters

    results = collection.query(**query_params)

    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "score": 1 - results["distances"][0][i],  # cosine similarity
        })

    return formatted


def get_collection_stats(collection_name: str = "earnings_transcripts") -> Dict:
    """Return stats about the collection."""
    collection = get_collection(collection_name)
    return {
        "name": collection_name,
        "count": collection.count(),
    }


def reset_collection(collection_name: str = "earnings_transcripts"):
    """Delete and recreate a collection (use with caution)."""
    global _collection
    client = _client or get_collection(collection_name)
    
    import chromadb
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/embeddings")
    client = chromadb.PersistentClient(path=persist_dir)
    
    try:
        client.delete_collection(collection_name)
        logger.warning(f"Deleted collection: {collection_name}")
    except Exception:
        pass
    
    _collection = None
    return get_collection(collection_name)
