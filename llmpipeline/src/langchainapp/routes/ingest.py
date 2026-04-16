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
from fastapi import APIRouter
import json
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from langsmith import traceable

from src.config.config import create_config
from src.langchainapp.services.vector_store_local import get_embeddings, upsert_documents, create_vector_store

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()
config_instance = create_config()

class IngestRequest(BaseModel):
    texts: list[str]
    metadata: list[dict]

@router.post("")
def ingest_documents(request: IngestRequest):
    return handler(request, None)

@traceable(name="ingest_doc", tags=["rag", "openai"])
def handler(event: IngestRequest, context) -> dict:
    try:

        texts: list[str] = event.texts
        metadata_list: list[dict] = event.metadata

        if not texts:
            return _resp(400, {"error": "Missing 'texts' in request body."})

        # Split each raw text into smaller chunks before embedding
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
        )
        documents = splitter.create_documents(texts, metadatas=metadata_list)

        embeddings = get_embeddings(config_instance.embedding_model)

        vs = create_vector_store(config_instance.embedding_model)
        upsert_documents(
            vs=vs,
            documents=documents
        )

        return _resp(
            200,
            {
                "message": (
                    f"Successfully ingested document with size: {len(texts)}"
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
