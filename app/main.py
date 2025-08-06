from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app.api import hackrx
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="Intelligent Query-Retrieval System")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API router
app.include_router(hackrx.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Intelligent Query-Retrieval System API."}
