"""
Vector store utilities for local development.
"""
from uuid import uuid4

from langchain_chroma import Chroma
import logging
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from src.config.config import create_config

logger = logging.getLogger(__name__)
config_instance = create_config()

# ── Embedding helper ──────────────────────────────────────────────────────────

def get_embeddings(model: str) -> OpenAIEmbeddings:
    print(f"Using apikey: {config_instance.openai_api_key}")
    return OpenAIEmbeddings(model=model, api_key=config_instance.openai_api_key)

# ── Create Vector Store ──────────────────────────────────────────────────────────

def create_vector_store(embedding_model: str) -> Chroma:
    return Chroma(collection_name="example_collection",
                  embedding_function=get_embeddings(embedding_model),
                  persist_directory="./chroma_db")

# ── Document ingestion ────────────────────────────────────────────────────────

def upsert_documents(vs: Chroma, documents: list[Document]):
    """
    Embed *documents* and add them to the OpenSearch index.
    The index is created automatically on first write if it does not exist.
    """
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vs.add_documents(documents, ids=uuids)
    logger.info("Upserted %d document(s) into index '%s'.", len(documents))

# ── Document deletion ────────────────────────────────────────────────────────

def delete_documents(vs: Chroma, uuid: str):
    vs.delete(id=uuid)


# ── Query Vector Store ────────────────────────────────────────────────────────

def query_by_vector(vs: Chroma, text: str) -> list[Document] :
    results = vs.similarity_search_by_vector(
        embedding=get_embeddings("text-embedding-3-small").embed_query(text),
    )
    return results["documents"]


def similarity_search(vs: Chroma, text: str, k: int = 4) -> list[Document]:
    results = vs.similarity_search_with_score(
        text,
        k=k,
        filter=None
    )
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
    return [res for res, _ in results]
