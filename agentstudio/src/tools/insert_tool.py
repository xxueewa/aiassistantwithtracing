import chromadb
from langchain_core.documents import Document
from services.vector_store_local import create_vector_store, upsert_documents
from langchain.tools import tool
from langsmith import traceable

@tool
@traceable(run_type="chain", name="rag_update")
def insert(query: str) -> str:
    """
    based on user's query, summarize and insert the company domain knowledge into the db
    e.g. company employee profiles, company trip policy
    """
    vs = create_vector_store()
    upsert_documents(vs, [
        Document(page_content=query, metadata={"source": "chat"}),
    ])