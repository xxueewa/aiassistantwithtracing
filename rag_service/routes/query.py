"""
query/app.py
------------
Lambda handler for POST /query

Expected request body:
  { "question": "What is LangSmith?" }

Response body:
  {
    "answer": "...",
    "source_documents": [{ "content": "...", "metadata": {...} }, ...]
  }
"""

import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ragcorelib.rag_chain import run_rag_chain


class QueryRequest(BaseModel):
    question: str

router = APIRouter()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@router.post("")
def query(event: QueryRequest):
    return handler(event)

def handler(event: QueryRequest) -> dict:
    try:

        question = event.question.strip()

        if not question:
            raise HTTPException(status_code=400, detail="Bad Request. Missing question in the request body.")

        result = run_rag_chain(question)
        return _resp(200, result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Internal Server Error. Unhandled error in query handler: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


def _resp(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "answer": body.get("answer", ""),
        "source_documents": body.get("source_documents", []),
    }
