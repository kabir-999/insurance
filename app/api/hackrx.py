import uuid
from fastapi import APIRouter, HTTPException
from app.api.schemas import HackRxRequest, HackRxResponse, Answer
from app.services.document_service import process_document
from app.services.pinecone_service import upsert_to_pinecone, query_pinecone, delete_namespace
from app.services.llm_service import get_answer_from_llm
import asyncio

router = APIRouter()

@router.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def run_submission(request: HackRxRequest):
    # Check API keys
    from app.core.config import PINECONE_API_KEY, GOOGLE_API_KEY
    if not PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="Pinecone API key not configured")
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Google API key not configured")
    
    # Use a single persistent namespace for all insurance documents
    namespace = "insurance-documents"
    print(f"DEBUG: Using persistent namespace: {namespace}")
    try:
        # Process the document synchronously for faster response
        text = process_document(request.documents)
        print(f"DEBUG: Extracted text length: {len(text) if text else 0}")
        print(f"DEBUG: First 200 chars: {text[:200] if text else 'None'}")
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the document.")
        
        # Improved chunking strategy for better context retrieval
        chunk_size = 1200  # Larger chunks for better context
        overlap = 200      # Overlap between chunks for continuity
        max_chunks = 30    # More chunks for better coverage
        
        # Create overlapping chunks
        text_chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                text_chunks.append(chunk)
            if len(text_chunks) >= max_chunks:
                break
        
        print(f"DEBUG: Created {len(text_chunks)} chunks")
        print(f"DEBUG: First chunk preview: {text_chunks[0][:100] if text_chunks else 'None'}...")
        print(f"DEBUG: Using namespace: {namespace}")
        
        # Process chunks in parallel for faster embedding and upsert
        upserted_count = await upsert_to_pinecone(namespace, text_chunks)
        print(f"DEBUG: Upserted {upserted_count} vectors to Pinecone")
        
        if upserted_count == 0:
            raise HTTPException(status_code=500, detail="Failed to process document chunks.")
        
        # Process questions in parallel
        tasks = [
            process_question(q, namespace)
            for q in request.questions
        ]
        answers = await asyncio.gather(*tasks)
        
        return HackRxResponse(answers=answers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Keep the persistent namespace - no cleanup needed
        print(f"DEBUG: Request completed, keeping persistent namespace: {namespace}")

async def process_question(question: str, namespace: str) -> Answer:
    """Process a single question and return an Answer."""
    relevant_chunks = await query_pinecone(namespace, question, top_k=5)
    print(f"DEBUG: Found {len(relevant_chunks)} chunks for question: {question}")
    print(f"DEBUG: Chunks: {relevant_chunks[:2] if relevant_chunks else 'None'}")
    
    context = "\n\n".join(relevant_chunks) if relevant_chunks else ""
    answer = await get_answer_from_llm(question, context)
    
    return Answer(
        question=question,
        answer=answer,
        context=context if context.strip() else None
    )

async def cleanup_namespace(namespace: str) -> None:
    """Clean up the Pinecone namespace after processing is complete."""
    try:
        # Use a single persistent namespace for all insurance documents
        namespace = "insurance-documents"
        print(f"DEBUG: Using persistent namespace: {namespace}")
    except Exception as e:
        print(f"Error cleaning up namespace {namespace}: {e}")