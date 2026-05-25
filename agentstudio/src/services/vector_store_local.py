"""
Vector store utilities for local development.
"""
from uuid import uuid4

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import logging
import os
from functools import cache
from pydantic.types import SecretStr

load_dotenv()
logger = logging.getLogger(__name__)
embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
chat_model = os.environ.get("CHAT_MODEL", "gpt-4o-mini")

_CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chroma_db"),
)

# ── Embedding helper ──────────────────────────────────────────────────────────

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=embedding_model)

# ── Create Vector Store ──────────────────────────────────────────────────────────

@cache
def create_vector_store() -> Chroma:
    return Chroma(
        collection_name="rag_collection",
        embedding_function=get_embeddings(),
        persist_directory=_CHROMA_DB_PATH,
    )


def reset_vector_store() -> None:
    create_vector_store.cache_clear()

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
        embedding=get_embeddings().embed_query(text),
    )
    return results


def similarity_search(vs: Chroma, text: str, k: int = 4) -> list[Document]:
    results = vs.similarity_search_with_score(
        text,
        k=k,
        filter=None
    )
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
    return [res for res, _ in results]
