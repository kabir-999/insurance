from pinecone import Pinecone, ServerlessSpec
from app.core.config import PINECONE_API_KEY
from app.services.embedding_service import get_embedding

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

def upsert_to_pinecone(namespace: str, text_chunks: list[str]):
    """Upserts text chunks and their embeddings into a specific namespace in the Pinecone index."""
    index = pc.Index(INDEX_NAME)
    vectors = []
    for i, chunk in enumerate(text_chunks):
        embedding = get_embedding(chunk)
        if embedding:
            vectors.append({
                'id': f'chunk_{i}',
                'values': embedding,
                'metadata': {'text': chunk}
            })

    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)

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
