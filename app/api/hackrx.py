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
        # Process the document asynchronously to avoid blocking
        text = await asyncio.get_event_loop().run_in_executor(
            None, process_document, request.documents
        )
        print(f"DEBUG: Extracted text length: {len(text) if text else 0}")
        print(f"DEBUG: First 200 chars: {text[:200] if text else 'None'}")
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the document.")
        
        # Deployment-optimized chunking strategy
        chunk_size = 1000  # Balanced chunk size for deployment
        overlap = 150      # Reasonable overlap for quality
        max_chunks = 40    # Balanced chunks for deployment
        
        # Create overlapping chunks
        text_chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size].strip()
            if chunk and len(chunk) > 50:  # Only add meaningful chunks
                text_chunks.append(chunk)
            if len(text_chunks) >= max_chunks:
                break
        
        print(f"DEBUG: Created {len(text_chunks)} chunks")
        if text_chunks:
            print(f"DEBUG: First chunk preview: {text_chunks[0][:100]}...")
            print(f"DEBUG: Last chunk preview: {text_chunks[-1][:100]}...")
        print(f"DEBUG: Using namespace: {namespace}")
        
        # Process chunks in parallel for faster embedding and upsert
        upsert_task = asyncio.create_task(upsert_to_pinecone(namespace, text_chunks))
        
        # Wait for upsert to complete
        upserted_count = await upsert_task
        print(f"DEBUG: Upserted {upserted_count} vectors to Pinecone")
        
        if upserted_count == 0:
            raise HTTPException(status_code=500, detail="Failed to process document chunks.")
        
        # Small wait for deployment environment stability
        await asyncio.sleep(1)
        
        # Process questions in parallel
        tasks = [
            process_question(q, namespace)
            for q in request.questions
        ]
        answers = await asyncio.gather(*tasks)
        
        return HackRxResponse(answers=answers)
        
    except Exception as e:
        print(f"DEBUG: Error in run_submission: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Keep the persistent namespace - no cleanup needed
        print(f"DEBUG: Request completed, keeping persistent namespace: {namespace}")

async def process_question(question: str, namespace: str) -> Answer:
    """Process a single question and return an Answer."""
    print(f"DEBUG: Processing question: {question}")
    relevant_chunks = await query_pinecone(namespace, question, top_k=5)  # Deployment-optimized chunks
    print(f"DEBUG: Found {len(relevant_chunks)} chunks for question: {question}")
    
    if relevant_chunks:
        print(f"DEBUG: First chunk preview: {relevant_chunks[0][:150]}...")
        print(f"DEBUG: Total context length: {sum(len(chunk) for chunk in relevant_chunks)}")
    else:
        print(f"DEBUG: No chunks found for question: {question}")
    
    # Join chunks with clear separators
    context = "\n\n--- Document Section ---\n\n".join(relevant_chunks) if relevant_chunks else ""
    
    # Get answer from LLM
    answer = await get_answer_from_llm(question, context)
    
    # Always include context for debugging (remove this line if you don't want context in response)
    final_context = context if context.strip() else None
    
    print(f"DEBUG: Answer generated: {answer[:100]}...")
    print(f"DEBUG: Context included: {'Yes' if final_context else 'No'}")
    
    return Answer(
        question=question,
        answer=answer,
        context=None  # Context removed as requested - only used internally for LLM
    )

async def cleanup_namespace(namespace: str) -> None:
    """Clean up the Pinecone namespace after processing is complete."""
    try:
        # Use a single persistent namespace for all insurance documents
        namespace = "insurance-documents"
        print(f"DEBUG: Using persistent namespace: {namespace}")
    except Exception as e:
        print(f"Error cleaning up namespace {namespace}: {e}")