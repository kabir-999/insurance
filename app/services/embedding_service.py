import asyncio
from typing import List
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

# Load sentence-transformers model (384-dim)
MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(MODEL_NAME)
_executor = ThreadPoolExecutor(max_workers=6)

async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generates 384-dim embeddings for a batch of texts using sentence-transformers all-MiniLM-L6-v2."""
    print(f"DEBUG: Generating embeddings for {len(texts)} texts using {MODEL_NAME} (384-dim)")
    if not texts:
        print("DEBUG: No texts provided for embedding")
        return []

    loop = asyncio.get_event_loop()
    def embed_batch(texts):
        embeddings = _model.encode(texts, show_progress_bar=False, batch_size=32)
        return embeddings.tolist()
    embeddings = await loop.run_in_executor(_executor, embed_batch, texts)
    if embeddings and len(embeddings) > 0:
        print(f"DEBUG: Returning {len(embeddings)} total embeddings with dim {len(embeddings[0])}")
    else:
        print("DEBUG: No embeddings returned or empty result.")
    return embeddings
    chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
    
    # Collect results from parallel processing
    for result in chunk_results:
        if isinstance(result, Exception):
            print(f"DEBUG: AGGRESSIVE chunk processing error: {result}")
            continue
        if result:
            all_embeddings.extend(result)
    
    print(f"DEBUG: Returning {len(all_embeddings)} total embeddings")
    return all_embeddings

def get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """Generates embedding for a single text."""
    if not text or not text.strip():
        return []
    
    try:
        result = asyncio.run(get_embeddings_batch([text], task_type))
        return result[0] if result else []
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return []
    return result[0] if result else []