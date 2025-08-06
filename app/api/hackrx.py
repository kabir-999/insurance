from fastapi import APIRouter, HTTPException, Request, status, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.api.schemas import HackRxRequest, HackRxResponse, Answer
from app.services.document_service import process_document
from app.services.pinecone_service import upsert_to_pinecone, query_pinecone, delete_namespace
from app.services.llm_service import get_answer_from_llm
import asyncio

router = APIRouter()

security = HTTPBearer()

# --- NEW ENDPOINT: /hackrx/run ---
@router.post(
    "/hackrx/run",
    summary="Run HackRx Query",
    description="Accepts JSON: {\"documents\": <url>, \"questions\": [<q1>, ...]}<br>Requires Bearer token auth (hardcoded). Returns answers as JSON.",
    tags=["HackRx"],
)
async def hackrx_run(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Accepts JSON: {"documents": <url>, "questions": [<q1>, ...]}
    Requires Bearer token auth (hardcoded). Returns answers as JSON.
    """
    # Auth check
    AUTH_TOKEN = "1d1090f9f6e5c68e19c277483330d79b5a157aaafd8fc73f58a1b333c5513fd4"
    token = credentials.credentials if credentials else None
    if token != AUTH_TOKEN:
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Unauthorized"})
    body = await request.json()
    documents = body.get("documents")
    questions = body.get("questions")
    if not documents or not isinstance(questions, list):
        return JSONResponse(status_code=400, content={"detail": "Invalid input format"})
    # Reuse run_submission logic
    class DummyRequest:
        pass
    dummy = DummyRequest()
    dummy.documents = documents
    dummy.questions = questions
    # Use same logic as run_submission
    return await run_submission(dummy)
    # Use same logic as run_submission
    return await run_submission(dummy)

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
        
        # AGGRESSIVE chunking strategy for sub-10s performance
        chunk_size = 1200  # Larger chunks for better context
        overlap = 200      # Better overlap for quality
        max_chunks = 50    # More chunks for comprehensive coverage
        
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
        
        # AGGRESSIVE parallel processing for maximum speed
        upsert_task = asyncio.create_task(upsert_to_pinecone(namespace, text_chunks))
        
        # Wait for upsert to complete with timeout
        try:
            upserted_count = await asyncio.wait_for(upsert_task, timeout=15.0)  # 15s timeout for upsert
        except asyncio.TimeoutError:
            print("DEBUG: Upsert timeout - proceeding with existing vectors")
            upserted_count = 0
        print(f"DEBUG: Upserted {upserted_count} vectors to Pinecone")
        
        if upserted_count == 0:
            raise HTTPException(status_code=500, detail="Failed to process document chunks.")
        
        # Minimal wait for AGGRESSIVE performance
        await asyncio.sleep(0.5)  # Reduced wait time
        
        # Process questions in parallel
        tasks = [
            process_question(q, namespace)
            for q in request.questions
        ]
        answers = await asyncio.gather(*tasks)
        
        # Exclude None values from the response
        response = HackRxResponse(answers=answers)
        return JSONResponse(
            content=response.dict(exclude_none=True),
            status_code=200
        )
        
    except Exception as e:
        print(f"DEBUG: Error in run_submission: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Keep the persistent namespace - no cleanup needed
        print(f"DEBUG: Request completed, keeping persistent namespace: {namespace}")

async def process_question(question: str, namespace: str) -> Answer:
    """Process a single question and return an Answer. Surfaces error details for debugging."""
    try:
        print(f"DEBUG: Processing question: {question}")
        relevant_chunks = await query_pinecone(namespace, question, top_k=8)  # AGGRESSIVE chunk retrieval
        print(f"DEBUG: Found {len(relevant_chunks)} chunks for question: {question}")
        if relevant_chunks:
            print(f"DEBUG: First chunk preview: {relevant_chunks[0][:150]}...")
            print(f"DEBUG: Total context length: {sum(len(chunk) for chunk in relevant_chunks)}")
        else:
            print(f"DEBUG: No chunks found for question: {question}")
        context = "\n\n--- Document Section ---\n\n".join(relevant_chunks) if relevant_chunks else ""
        answer = await get_answer_from_llm(question, context)
        final_context = context if context.strip() else None
        print(f"DEBUG: Answer generated: {answer[:100]}...")
        print(f"DEBUG: Context included: {'Yes' if final_context else 'No'}")
        return Answer(
            question=question,
            answer=answer,
            context=None
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"PROCESS_QUESTION ERROR: {e}\n{tb}")
        return Answer(
            question=question,
            answer=f"PROCESS_QUESTION ERROR: {e}\n{tb}",
            context=None
        )

async def cleanup_namespace(namespace: str) -> None:
    """Clean up the Pinecone namespace after processing is complete."""
    try:
        # Use a single persistent namespace for all insurance documents
        namespace = "insurance-documents"
        print(f"DEBUG: Using persistent namespace: {namespace}")
    except Exception as e:
        print(f"Error cleaning up namespace {namespace}: {e}")