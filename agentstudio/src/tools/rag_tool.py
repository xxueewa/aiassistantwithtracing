from langchain.tools import tool
from langsmith import traceable

from ragcorelib import run_rag_chain

@tool
@traceable(run_type="chain", name="rag_chain")
def info_retrieval(query: str) -> str:
    """
    retrieves info from rag service
    """
    answer = run_rag_chain(query)
    return answer["answer"]