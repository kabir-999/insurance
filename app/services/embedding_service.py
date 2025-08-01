import asyncio
from typing import List, Optional
import google.generativeai as genai
from app.core.config import GOOGLE_API_KEY
from concurrent.futures import ThreadPoolExecutor

# Configure the generative AI model
genai.configure(api_key=GOOGLE_API_KEY)

# Thread pool for parallel processing
_executor = ThreadPoolExecutor(max_workers=4)

async def get_embeddings_batch(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """Generates embeddings for a batch of texts using Google's model with parallel processing."""
    if not texts:
        return []

    # Process in chunks to avoid rate limits
    chunk_size = 50
    all_embeddings = []
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        try:
            # Run in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda c=chunk: genai.embed_content(
                    model="models/text-embedding-004",
                    content=c,
                    task_type=task_type
                )
            )
            if result and 'embedding' in result:
                all_embeddings.extend(result['embedding'])
        except Exception as e:
            print(f"Error in embedding batch {i//chunk_size}: {e}")
            # Return partial results if some embeddings succeeded
            if all_embeddings:
                return all_embeddings
            return []
    
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