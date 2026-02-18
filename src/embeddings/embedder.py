"""
Embedding generator using Sentence Transformers.
Generates dense vector representations for transcript chunks
and stores them in ChromaDB.
"""
import os
from typing import List, Dict, Optional
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Lazy imports for heavy libraries
_model = None
_model_name = None


def _get_model(model_name: str = None):
    """Lazy-load the sentence transformer model."""
    global _model, _model_name
    model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    
    if _model is None or _model_name != model_name:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        _model_name = model_name
        logger.info(f"Model loaded. Embedding dimension: {_model.get_sentence_embedding_dimension()}")
    
    return _model


def generate_embeddings(texts: List[str], model_name: str = None,
                        batch_size: int = 64, show_progress: bool = True) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of text strings to embed.
        model_name: HuggingFace model identifier.
        batch_size: Batch size for encoding.
        show_progress: Show progress bar.

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    model = _get_model(model_name)
    
    logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )
    
    return embeddings.tolist()


def embed_chunks(chunks: List[Dict], model_name: str = None,
                 batch_size: int = 64) -> List[Dict]:
    """
    Add embeddings to chunk dictionaries.

    Args:
        chunks: List from chunker.chunk_segments() — each has "text" and "metadata".

    Returns:
        Same chunks with "embedding" field added.
    """
    texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(texts, model_name, batch_size)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    logger.info(f"Embedded {len(chunks)} chunks (dim={len(embeddings[0])})")
    return chunks


def get_embedding_dimension(model_name: str = None) -> int:
    """Return the embedding dimension for the configured model."""
    model = _get_model(model_name)
    return model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    # Quick test
    test_texts = [
        "Apple reported strong revenue growth driven by AI platform adoption.",
        "Supply chain disruptions continue to impact manufacturing costs.",
        "The company announced a $10 billion share repurchase program.",
    ]
    embeddings = generate_embeddings(test_texts)
    print(f"Generated {len(embeddings)} embeddings, dim={len(embeddings[0])}")
    
    # Test similarity
    import numpy as np
    sim_01 = np.dot(embeddings[0], embeddings[1])
    sim_02 = np.dot(embeddings[0], embeddings[2])
    print(f"Similarity (revenue vs supply chain): {sim_01:.4f}")
    print(f"Similarity (revenue vs buyback): {sim_02:.4f}")
