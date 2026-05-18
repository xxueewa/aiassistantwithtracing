from langchain.tools import tool
from tavily import TavilyClient
import os

tavily_api_key = str(os.environ["TAVILY_API_KEY"])
tavily_client = TavilyClient(api_key=tavily_api_key)

@tool
def search(query: str) -> str:
    """
    use Tavily package to search for information on the web
    TODO: develop customized search tool
    """
    response = tavily_client.search(
        query=query,
        include_answer="basic"
    )
    return response["answer"]