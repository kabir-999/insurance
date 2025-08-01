import google.generativeai as genai
from app.core.config import GOOGLE_API_KEY

# Configure the generative AI model
genai.configure(api_key=GOOGLE_API_KEY)

def get_embeddings_batch(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Generates embeddings for a batch of texts using Google's model."""
    try:
        result = genai.embed_content(model="models/text-embedding-004",
                                     content=texts,
                                     task_type=task_type)
        return result['embedding']
    except Exception as e:
        print(f"An error occurred during batch embedding: {e}")
        return []

def get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT"):
    """Generates embedding for the given text using Google's model."""
    if not text.strip():
        print("Attempted to embed empty or whitespace text. Skipping.")
        return None
    try:
        # Using the text-embedding-004 model, which is a common choice for this task
        result = genai.embed_content(model="models/text-embedding-004",
                                     content=text,
                                     task_type=task_type)
        return result['embedding']
    except Exception as e:
        print(f"An error occurred during embedding: {e}")
        return None
