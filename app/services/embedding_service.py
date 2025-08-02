import asyncio
from typing import List, Optional
import google.generativeai as genai
from app.core.config import GOOGLE_API_KEY
from concurrent.futures import ThreadPoolExecutor

# Configure the generative AI model
genai.configure(api_key=GOOGLE_API_KEY)

# Thread pool AGGRESSIVELY optimized for deployment environment
_executor = ThreadPoolExecutor(max_workers=12)  # AGGRESSIVE workers for speed

# Connection pooling for Google API
_api_client_cache = None

async def get_embeddings_batch(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """Generates embeddings for a batch of texts using Google's model with parallel processing."""
    print(f"DEBUG: Generating embeddings for {len(texts)} texts (task_type: {task_type})")
    if not texts:
        print("DEBUG: No texts provided for embedding")
        return []

    # Process in AGGRESSIVE chunks for maximum speed
    chunk_size = 200  # AGGRESSIVE chunk size for speed
    all_embeddings = []
    
    # Process all chunks in parallel for maximum speed
    async def process_chunk_async(chunk_texts, chunk_idx):
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: genai.embed_content(
                    model="models/text-embedding-004",
                    content=chunk_texts,
                    task_type=task_type
                )
            )
            if result and 'embedding' in result:
                batch_embeddings = result['embedding']
                print(f"DEBUG: AGGRESSIVE chunk {chunk_idx} completed: {len(batch_embeddings)} embeddings")
                return batch_embeddings
            else:
                print(f"DEBUG: No embeddings in result for AGGRESSIVE chunk {chunk_idx}: {result}")
                return []
        except Exception as e:
            print(f"Error in AGGRESSIVE embedding chunk {chunk_idx}: {e}")
            return []
    
    # Create all chunk tasks
    chunk_tasks = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        task = process_chunk_async(chunk, i//chunk_size)
        chunk_tasks.append(task)
    
    # Execute all chunk tasks in parallel
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