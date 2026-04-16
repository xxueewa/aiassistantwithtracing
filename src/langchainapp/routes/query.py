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

import json
import logging
from fastapi import APIRouter
from pydantic import BaseModel

from src.config.config import create_config
from src.langchainapp.services.rag_chain import run_rag_chain


class QueryRequest(BaseModel):
    question: str

router = APIRouter()
config_instance = create_config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@router.post("")
def query(event: QueryRequest):
    return handler(event)

def handler(event: QueryRequest) -> dict:
    try:

        question = event.question.strip()

        if not question:
            return _resp(400, {"error": "Missing 'question' in request body."})

        result = run_rag_chain(question)
        return _resp(200, result)

    except Exception as exc:
        logger.exception("Unhandled error in query handler: %s", exc)
        return _resp(500, {"error": "Internal server error."})


def _resp(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "answer": body.get("answer", ""),
        "source_documents": body.get("source_documents", []),
    }
