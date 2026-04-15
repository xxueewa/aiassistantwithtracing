"""
Vector store utilities for local development.
"""
from uuid import uuid4

from langchain_chroma import Chroma
import logging
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# ── Embedding helper ──────────────────────────────────────────────────────────

def get_embeddings(model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model)

# ── Create Vector Store ──────────────────────────────────────────────────────────

def create_vector_store() -> Chroma:
    return Chroma(collection_name="example_collection",
                  embedding_function=get_embeddings("text-embedding-3-small"),
                  persist_directory="./chroma_db")

# ── Document ingestion ────────────────────────────────────────────────────────

def upsert_documents(vs: Chroma, documents: list[Document]):
    """
    Embed *documents* and add them to the OpenSearch index.
    The index is created automatically on first write if it does not exist.
    """
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vs.add_documents(documents, ids=uuids)
    logger.info("Upserted %d document(s) into index '%s'.", len(documents), vs.collection_name)

# ── Document deletion ────────────────────────────────────────────────────────

def delete_documents(vs: Chroma, uuid: str):
    vs.delete(id=uuid)