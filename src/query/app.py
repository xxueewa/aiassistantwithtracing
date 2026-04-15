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

from config.config import get_config, setup_langsmith
from services.rag_chain import run_rag_chain

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_config = None


def _get_config():
    global _config
    if _config is None:
        _config = get_config()
        setup_langsmith(_config)
    return _config


def handler(event: dict, context) -> dict:
    try:
        config = _get_config()
        body = json.loads(event.get("body") or "{}")
        question = body.get("question", "").strip()

        if not question:
            return _resp(400, {"error": "Missing 'question' in request body."})

        result = run_rag_chain(question, config)
        return _resp(200, result)

    except Exception as exc:
        logger.exception("Unhandled error in query handler: %s", exc)
        return _resp(500, {"error": "Internal server error."})


def _resp(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
