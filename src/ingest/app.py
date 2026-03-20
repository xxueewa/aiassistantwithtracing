"""
ingest/app.py
-------------
Lambda handler for POST /ingest

Expected request body:
  {
    "texts": ["doc text 1", "doc text 2", ...],
    "metadata": [{"source": "wiki"}, {"source": "pdf"}, ...]   // optional
  }

Response body:
  { "message": "...", "chunks_added": 12 }
"""

import json
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_config, setup_langsmith
from vector_store import get_embeddings, upsert_documents

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

        texts: list[str] = body.get("texts", [])
        metadata_list: list[dict] = body.get("metadata", [{}] * len(texts))

        if not texts:
            return _resp(400, {"error": "Missing 'texts' in request body."})

        # Split each raw text into smaller chunks before embedding
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
        )
        documents = splitter.create_documents(texts, metadatas=metadata_list)

        embeddings = get_embeddings(config.embedding_model)
        upsert_documents(
            documents=documents,
            endpoint=config.opensearch_endpoint,
            index_name=config.opensearch_index,
            embeddings=embeddings,
            region=config.aws_region,
        )

        return _resp(
            200,
            {
                "message": (
                    f"Successfully ingested {len(texts)} document(s) "
                    f"as {len(documents)} chunk(s)."
                ),
                "chunks_added": len(documents),
            },
        )

    except Exception as exc:
        logger.exception("Unhandled error in ingest handler: %s", exc)
        return _resp(500, {"error": "Internal server error."})


def _resp(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
