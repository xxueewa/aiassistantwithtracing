from langchain.tools import tool
from langsmith import traceable

from rag_core.rag_chain import run_rag_chain

@tool
@traceable(run_type="chain", name="rag_chain")
def info_retrieval(query: str) -> str:
    answer = run_rag_chain(query)
    return answer["answer"]