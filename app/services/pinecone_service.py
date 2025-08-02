import asyncio
from typing import List, Optional
import pinecone
from pinecone import Pinecone, ServerlessSpec
from app.core.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
from app.services.embedding_service import get_embeddings_batch
import asyncio
from typing import List, Optional
import concurrent.futures

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
# Use index name from environment variables
INDEX_NAME = PINECONE_INDEX_NAME

# AWS region that supports free tier
AWS_REGION = "us-east-1"  # us-east-1 supports free tier

# Configure for better context processing
BATCH_SIZE = 15  # Balanced batch size for good processing
MAX_WORKERS = 3  # Moderate workers for efficiency

async def create_pinecone_index():
    """Creates an optimized Pinecone index if it doesn't exist.
    
    Using AWS us-east-1 which supports the free tier.
    """
    try:
        # List indexes is a synchronous operation, so we run it in a thread
        indexes = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pc.list_indexes().names()
        )
        
        if INDEX_NAME in indexes:
            # Check if existing index has correct dimensions
            try:
                index_info = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: pc.describe_index(INDEX_NAME)
                )
                if index_info.dimension != 768:
                    print(f"Index {INDEX_NAME} has wrong dimensions ({index_info.dimension}), deleting and recreating...")
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: pc.delete_index(INDEX_NAME)
                    )
                    # Wait a bit for deletion to complete
                    await asyncio.sleep(10)
                    indexes = []  # Force recreation
                else:
                    print(f"Index {INDEX_NAME} already exists with correct dimensions")
                    return True
            except Exception as e:
                print(f"Error checking index dimensions: {e}")
                # Continue to recreation if there's an issue
        
        if INDEX_NAME not in indexes:
            print(f"Creating new index: {INDEX_NAME} in {AWS_REGION}")
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pc.create_index(
                    name=INDEX_NAME,
                    dimension=768,  # Dimension for text-embedding-004 model
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=AWS_REGION
                    ),
                    timeout=30
                )
            )
            print(f"Successfully created index: {INDEX_NAME}")
        return True
    except Exception as e:
        print(f"Error in create_pinecone_index: {e}")
        print("Make sure your account has access to create indexes in the free tier.")
        raise

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
    print(f"DEBUG: Starting upsert for {len(text_chunks)} chunks to namespace '{namespace}'")
    await create_pinecone_index()
    loop = asyncio.get_event_loop()
    
    # Create index instance in the event loop
    index = await loop.run_in_executor(None, lambda: pc.Index(INDEX_NAME))
    
    # Filter and prepare chunks with indices
    valid_chunks = [(i, chunk) for i, chunk in enumerate(text_chunks) if chunk and chunk.strip()]
    if not valid_chunks:
        print(f"DEBUG: No valid chunks to upsert")
        return 0
    
    print(f"DEBUG: Processing {len(valid_chunks)} valid chunks")
    
    # Process in parallel batches
    total_vectors = 0
    for i in range(0, len(valid_chunks), BATCH_SIZE):
        batch = valid_chunks[i:i + BATCH_SIZE]
        print(f"DEBUG: Processing batch {i//BATCH_SIZE + 1} with {len(batch)} chunks")
        
        vectors = await process_batch(batch)
        
        if vectors:
            print(f"DEBUG: Generated {len(vectors)} vectors for batch")
            # Upsert in smaller batches for reliability
            for j in range(0, len(vectors), 10):
                batch_vectors = vectors[j:j+10]
                try:
                    # Run upsert in thread pool with reasonable timeout
                    await loop.run_in_executor(
                        None,
                        lambda v=batch_vectors: index.upsert(
                            vectors=v,
                            namespace=namespace,
                            timeout=10  # Reasonable timeout for reliability
                        )
                    )
                    print(f"DEBUG: Successfully upserted {len(batch_vectors)} vectors")
                except Exception as e:
                    print(f"DEBUG: Error upserting batch: {e}")
                    # Continue with other batches even if one fails
                    continue
            total_vectors += len(vectors)
        else:
            print(f"DEBUG: No vectors generated for batch {i//BATCH_SIZE + 1}")
    
    print(f"DEBUG: Total vectors upserted: {total_vectors}")
    return total_vectors

async def query_pinecone(namespace: str, query: str, top_k: int = 5) -> List[str]:
    """Optimized query with faster response time."""
    print(f"DEBUG: Querying namespace '{namespace}' with query: '{query}' (top_k={top_k})")
    await create_pinecone_index()
    loop = asyncio.get_event_loop()
    
    # Create index instance in the event loop
    index = await loop.run_in_executor(None, lambda: pc.Index(INDEX_NAME))
    
    try:
        # Check if namespace has any vectors
        try:
            stats = await loop.run_in_executor(
                None,
                lambda: index.describe_index_stats()
            )
            namespace_stats = stats.namespaces.get(namespace, None)
            if namespace_stats:
                print(f"DEBUG: Namespace '{namespace}' has {namespace_stats.vector_count} vectors")
            else:
                print(f"DEBUG: Namespace '{namespace}' not found in index stats")
                # Try to list all namespaces
                all_namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
                print(f"DEBUG: Available namespaces: {all_namespaces}")
        except Exception as e:
            print(f"DEBUG: Could not get namespace stats: {e}")
        
        # Get embedding with timeout
        print(f"DEBUG: Getting embedding for query...")
        query_embedding = await get_embeddings_batch([query], task_type="RETRIEVAL_QUERY")
        if not query_embedding:
            print(f"DEBUG: Failed to get query embedding")
            return []
        
        print(f"DEBUG: Got query embedding with dimension: {len(query_embedding[0])}")
            
        # Run query in thread pool for better context
        print(f"DEBUG: Executing Pinecone query...")
        results = await loop.run_in_executor(
            None,
            lambda: index.query(
                vector=query_embedding[0],
                top_k=min(top_k, 10),  # More results for better context
                include_metadata=True,
                namespace=namespace,
                timeout=15  # Longer timeout for better results
            )
        )
        
        print(f"DEBUG: Query returned {len(results.matches)} matches")
        if len(results.matches) == 0:
            print(f"DEBUG: No matches found. This could be due to:")
            print(f"DEBUG: 1. Namespace mismatch (using '{namespace}')")
            print(f"DEBUG: 2. Embedding similarity too low")
            print(f"DEBUG: 3. Index not properly populated")
            print(f"DEBUG: 4. Vectors still being indexed (try waiting longer)")
            return []
        
        # Filter matches with reasonable similarity scores
        good_matches = [match for match in results.matches if match.score > 0.3]
        print(f"DEBUG: Found {len(good_matches)} matches with score > 0.3")
        
        for i, match in enumerate(results.matches[:5]):
            text_preview = match.metadata.get('text', '')[:100] if match.metadata else 'No metadata'
            print(f"DEBUG: Match {i+1} score: {match.score:.4f}, text: {text_preview}...")
        
        # Return matches with decent scores, but include all if none are good
        matches_to_use = good_matches if good_matches else results.matches
        texts = [match.metadata.get("text", "") for match in matches_to_use if match.metadata]
        
        # Filter out empty texts
        texts = [text for text in texts if text and text.strip()]
        
        print(f"DEBUG: Returning {len(texts)} text chunks")
        return texts
    except Exception as e:
        print(f"DEBUG: Query error: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        return []

async def delete_namespace(namespace: str) -> None:
    """Delete all vectors in the specified namespace."""
    try:
        loop = asyncio.get_event_loop()
        index = await loop.run_in_executor(None, lambda: pc.Index(INDEX_NAME))
        await loop.run_in_executor(
            None,
            lambda: index.delete(delete_all=True, namespace=namespace)
        )
        print(f"Successfully deleted namespace: {namespace}")
    except Exception as e:
        print(f"Error deleting namespace {namespace}: {e}")
        raise