from typing import List, Optional
import pinecone
from pinecone import Pinecone, ServerlessSpec
from app.core.config import PINECONE_API_KEY
from app.services.embedding_service import get_embeddings_batch
import asyncio
from typing import List, Optional
import concurrent.futures

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "hackrx-index-optimized"  # New index name to avoid conflicts

# Configure for better performance
BATCH_SIZE = 50  # Number of chunks to process in parallel
MAX_WORKERS = 4  # Number of worker threads

def create_pinecone_index():
    """Creates an optimized Pinecone index if it doesn't exist."""
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,  # Adjust based on your embedding model
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            ),
            timeout=30  # Shorter timeout for faster failure
        )

async def process_batch(batch: List[tuple[int, str]]) -> List[dict]:
    """Process a batch of chunks into vectors."""
    indices, chunks = zip(*batch)
    embeddings = await get_embeddings_batch(list(chunks))
    if not embeddings:
        return []
    
    return [
        {
            'id': f'chunk_{idx}',
            'values': emb,
            'metadata': {'text': chunk}
        }
        for idx, chunk, emb in zip(indices, chunks, embeddings)
    ]

async def upsert_to_pinecone(namespace: str, text_chunks: List[str]) -> int:
    """Upserts text chunks in parallel for better performance."""
    await create_pinecone_index()
    index = pc.Index(INDEX_NAME)
    
    # Filter and prepare chunks with indices
    valid_chunks = [(i, chunk) for i, chunk in enumerate(text_chunks) if chunk and chunk.strip()]
    if not valid_chunks:
        return 0
    
    # Process in parallel batches
    total_vectors = 0
    for i in range(0, len(valid_chunks), BATCH_SIZE):
        batch = valid_chunks[i:i + BATCH_SIZE]
        vectors = await process_batch(batch)
        
        if vectors:
            # Upsert in smaller batches to avoid timeouts
            for j in range(0, len(vectors), 50):
                index.upsert(
                    vectors=vectors[j:j+50],
                    namespace=namespace,
                    timeout=10  # Shorter timeout for faster retries
                )
            total_vectors += len(vectors)
    
    return total_vectors

async def query_pinecone(namespace: str, query: str, top_k: int = 5) -> List[str]:
    """Optimized query with faster response time."""
    await create_pinecone_index()
    index = pc.Index(INDEX_NAME)
    
    # Get embedding with timeout
    try:
        query_embedding = await get_embeddings_batch([query], task_type="RETRIEVAL_QUERY")
        if not query_embedding:
            return []
            
        results = index.query(
            vector=query_embedding[0],
            top_k=min(top_k, 5),  # Limit to 5 for faster response
            include_metadata=True,
            namespace=namespace,
            timeout=5  # Faster timeout for queries
        )
        
        return [match.metadata.get("text", "") for match in results.matches]
    except Exception as e:
        print(f"Query error: {e}")
        return []