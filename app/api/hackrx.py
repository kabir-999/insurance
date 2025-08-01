import uuid
import time
from fastapi import APIRouter, Depends, HTTPException, Request
from app.api.schemas import HackRxRequest, HackRxResponse, Answer
from app.services.document_service import process_document
from app.services.pinecone_service import create_pinecone_index, upsert_to_pinecone, query_pinecone, delete_namespace
from app.services.llm_service import get_answer_from_llm

router = APIRouter()

async def verify_token(request: Request):
    authorization = request.headers.get("Authorization")
    if not authorization or authorization != "Bearer 1d1090f9f6e5c68e19c277483330d79b5a157aaafd8fc73f58a1b333c5513fd4":
        raise HTTPException(status_code=401, detail="Unauthorized")

@router.post("/api/v1/hackrx/run", response_model=HackRxResponse, dependencies=[Depends(verify_token)])
def run_submission(request: HackRxRequest):
    namespace = f"hackrx-namespace-{uuid.uuid4().hex}"
    try:
        # 1. Create the single persistent index if it doesn't exist.
        # This is idempotent and safe to call on every request.
        create_pinecone_index()

        # 2. Process the document
        text = process_document(request.documents)
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the document.")
        
        # A simple chunking strategy
        text_chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]

        # 3. Upsert data into the new namespace
        upsert_to_pinecone(namespace, text_chunks)

        # Add a 10-second delay to allow for Pinecone indexing latency
        time.sleep(10)

        # 4. For each question, query Pinecone and get answer from LLM
        answers = []
        for q in request.questions:
            context = query_pinecone(namespace, q, top_k=7)
            answer_text = get_answer_from_llm(q, context)
            answers.append(Answer(question=q, answer=answer_text))
        
        return HackRxResponse(answers=answers)

    except Exception as e:
        # Log the exception for debugging
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 5. Clean up the namespace from the Pinecone index to keep it clean
        if 'namespace' in locals():
            delete_namespace(namespace)
