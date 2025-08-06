from pydantic import BaseModel, HttpUrl
from typing import List

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: str
    context: str | None = None

    class Config:
        json_encoders = {
            'Answer': lambda v: v.dict(exclude_none=True)
        }

class HackRxResponse(BaseModel):
    answers: List[Answer]
