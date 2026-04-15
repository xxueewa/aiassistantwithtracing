"""
config.py
---------
Reads runtime configuration from environment variables and activates
LangSmith tracing. All secrets are injected by Lambda from SSM SecureString
parameters — never hard-coded here.
"""

import os
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Config:
    # OpenAI
    openai_api_key: str
    # LangSmith
    langchain_api_key: str
    langchain_project: str
    langchain_tracing_v2: bool
    langchain_endpoint: str
    # OpenSearch
    opensearch_endpoint: str   # host only, no https:// prefix
    opensearch_index: str
    aws_region: str
    # Models
    embedding_model: str
    chat_model: str

config_instance = Config(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        langchain_api_key=os.environ.get("LANGCHAIN_API_KEY", ""),
        langchain_project=os.environ.get("LANGCHAIN_PROJECT", "langsmith-rag-demo"),
        langchain_tracing_v2=os.environ.get("LANGCHAIN_TRACING_V2", "true").lower() == "true",
        langchain_endpoint=os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        opensearch_endpoint=os.environ.get("OPENSEARCH_ENDPOINT", ""),
        opensearch_index=os.environ.get("OPENSEARCH_INDEX", "rag-documents"),
        aws_region=os.environ.get("AWS_REGION", "us-east-1"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.environ.get("CHAT_MODEL", "gpt-4o-mini"),
    )