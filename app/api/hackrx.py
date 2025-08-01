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
    namespace = f"hackrx-namespace-{uuid.uuid4().hex}"
    try:
        # Process the document
        text = process_document(request.documents)
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the document.")
        
        # Chunk the text
        chunk_size = 1000  # Optimized chunking strategy with smaller chunks for better parallel processing
        text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Process chunks in parallel for faster embedding and upsert
        upserted_count = await upsert_to_pinecone(namespace, text_chunks)
        
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
        # Clean up namespace in the background
        asyncio.create_task(cleanup_namespace(namespace))

async def process_question(question: str, namespace: str) -> Answer:
    """Process a single question and return an Answer."""
    relevant_chunks = await query_pinecone(namespace, question, top_k=3)
    context = "\n\n".join(relevant_chunks)
    answer = await get_answer_from_llm(question, context)
    return Answer(
        question=question,
        answer=answer,
        context=context if context else None
    )

async def cleanup_namespace(namespace: str) -> None:
    """Clean up the Pinecone namespace after processing is complete."""
    try:
        if namespace:
            await delete_namespace(namespace)
    except Exception as e:
        print(f"Error cleaning up namespace {namespace}: {e}")