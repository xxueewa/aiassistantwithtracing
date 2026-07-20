from langchain.tools import tool
from langsmith import traceable

from services.rag_chain import run_rag_chain

@tool
@traceable(run_type="chain", name="rag_chain")
def info_retrieval(query: str) -> str:
    """
    retrieves domain info from db
    e.g. company employee profiles, meeting agenda, calendar info
    """
    import requests
    url = "http://0.0.0.0:80/query"
    response = requests.post(url, json={"question": query})

    if response.status_code == 200:
        answer = response.json()
        return answer["answer"]

    return {"error": "Could not fetch data", "status_code": response.status_code}