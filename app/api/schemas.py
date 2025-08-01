from pydantic import BaseModel, HttpUrl
from typing import List

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: str

class HackRxResponse(BaseModel):
    answers: List[Answer]
