from fastapi import FastAPI
from app.api import hackrx
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="Intelligent Query-Retrieval System")

# Include the API router
app.include_router(hackrx.router, prefix="/api/v1/hackrx")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Intelligent Query-Retrieval System API."}
