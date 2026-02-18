"""
Import Script — Load Kaggle processed data (Customer Voice) into local ChromaDB.
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.vector_store import get_collection

def import_data(
    parquet_path: str = "data/raw/all_chunks.parquet",
    npy_path: str = "data/raw/embeddings.npy"
):
    """
    Import pre-computed embeddings and chunks into ChromaDB.
    Handles 'clothing_reviews' and 'twitter_support' schemas.
    """
    if not os.path.exists(parquet_path) or not os.path.exists(npy_path):
        logger.error(f"Files not found: {parquet_path} or {npy_path}")
        logger.info("Please download 'all_chunks.parquet' and 'embeddings.npy' from Kaggle and place them in 'data/raw/'")
        return

    logger.info("Loading data from disk...")
    df = pd.read_parquet(parquet_path)
    embeddings = np.load(npy_path)

    if len(df) != len(embeddings):
        logger.error(f"Mismatch: {len(df)} chunks vs {len(embeddings)} embeddings")
        return

    collection = get_collection()
    logger.info(f"Importing {len(df)} documents into ChromaDB...")

    batch_size = 500
    ids = []
    metadatas = []
    documents = []
    batch_embeddings = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        # Generate unique ID
        chunk_id = f"cv_{i}"
        
        # Prepare metadata (Handle both schemas + common fields)
        # Common fields: source, topic (if available)
        meta = {
            "source": str(row.get("source", "unknown")),
            "topic": int(row.get("topic", -1)),
        }
        
        # Clothing Reviews specific
        if row.get("source") == "clothing_reviews":
            meta.update({
                "rating": int(row.get("rating", 0)),
                "category": str(row.get("category", "Unknown")),
                "age": int(row.get("age", 0)),
                "recommended": int(row.get("recommended", 0)),
                "sizing_feedback": str(row.get("sizing_feedback", "Neutral")), 
            })
            
        # Twitter Support specific
        elif row.get("source") == "twitter_support":
            meta.update({
                "author": str(row.get("author", "unknown")),
                "issue_type": str(row.get("issue_type", "General")),
                "confidence": float(row.get("confidence", 0.0)),
            })
            
        # Clean up any NaN/Nulls in metadata values to avoid ChromaDB errors
        clean_meta = {}
        for k, v in meta.items():
            if pd.isna(v):
                clean_meta[k] = "Unknown" if isinstance(v, str) else 0
            else:
                clean_meta[k] = v

        ids.append(chunk_id)
        documents.append(row["text"])
        metadatas.append(clean_meta)
        batch_embeddings.append(embeddings[i].tolist())

        # Upsert batch
        if len(ids) >= batch_size:
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings,
                metadatas=metadatas
            )
            ids, documents, metadatas, batch_embeddings = [], [], [], []

    # Final batch
    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=batch_embeddings,
            metadatas=metadatas
        )

    logger.info(f"Import complete. Total docs in collection: {collection.count()}")

if __name__ == "__main__":
    import_data()
