from pinecone import Pinecone, ServerlessSpec
from app.core.config import PINECONE_API_KEY
from app.services.embedding_service import get_embedding, get_embeddings_batch

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "insurance-query-system"

def create_pinecone_index(dimension: int = 768):
    """Creates a single, persistent serverless Pinecone index if it does not exist."""
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1' # Standard free tier region
            )
        )

def upsert_to_pinecone(namespace: str, text_chunks: list[str]) -> int:
    """Upserts text chunks and their embeddings into a specific namespace in the Pinecone index."""
    index = pc.Index(INDEX_NAME)

    # Filter out empty or whitespace-only chunks
    valid_chunks = [chunk for chunk in text_chunks if chunk and chunk.strip()]
    if not valid_chunks:
        return 0

    embeddings = get_embeddings_batch(valid_chunks)
    if not embeddings or len(embeddings) != len(valid_chunks):
        print("Error: Mismatch between number of chunks and embeddings returned.")
        return 0

    vectors = [
        {'id': f'chunk_{i}', 'values': emb, 'metadata': {'text': chunk}}
        for i, (chunk, emb) in enumerate(zip(valid_chunks, embeddings))
    ]

    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)
        return len(vectors)
    return 0

def get_namespace_vector_count(namespace: str) -> int:
    """Gets the number of vectors in a specific namespace."""
    try:
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        return stats.namespaces.get(namespace, {}).get('vector_count', 0)
    except Exception as e:
        print(f"Error fetching namespace stats: {e}")
        return 0

def query_pinecone(namespace: str, query: str, top_k: int = 5) -> list[str]:
    """Queries a namespace in the Pinecone index and returns the most relevant text chunks."""
    index = pc.Index(INDEX_NAME)
    query_embedding = get_embedding(query, task_type="RETRIEVAL_QUERY")
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace=namespace)
    return [match['metadata']['text'] for match in results['matches']]

def delete_namespace(namespace: str):
    """Deletes a namespace and all its vectors from the index."""
    index = pc.Index(INDEX_NAME)
    index.delete(namespace=namespace, delete_all=True)

def delete_index(index_name: str):
    """Deletes a Pinecone index."""
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
