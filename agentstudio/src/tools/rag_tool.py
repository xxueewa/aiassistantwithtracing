from langchain.tools import tool
from langsmith import traceable

from services.rag_chain import run_rag_chain

@tool
@traceable(run_type="chain", name="rag_chain")
def info_retrieval(query: str) -> str:
    """
    retrieves domain info from rag service
    e.g. company employee profiles
    """
    answer = run_rag_chain(query)
    return answer["answer"]