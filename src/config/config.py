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


def get_config() -> Config:
    """Build Config from environment variables. Raises if required vars are missing."""
    return Config(
        openai_api_key=_require("OPENAI_API_KEY"),
        langchain_api_key=_require("LANGCHAIN_API_KEY"),
        langchain_project=os.environ.get("LANGCHAIN_PROJECT", "langsmith-rag-demo"),
        langchain_tracing_v2=os.environ.get("LANGCHAIN_TRACING_V2", "true").lower() == "true",
        langchain_endpoint=os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        opensearch_endpoint=_require("OPENSEARCH_ENDPOINT"),
        opensearch_index=os.environ.get("OPENSEARCH_INDEX", "rag-documents"),
        aws_region=os.environ.get("AWS_REGION", "us-east-1"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.environ.get("CHAT_MODEL", "gpt-4o-mini"),
    )


def setup_langsmith(config: Config) -> None:
    """
    Propagate LangSmith settings to environment variables so that LangChain
    auto-tracing picks them up transparently — no extra instrumentation needed.

    With LANGCHAIN_TRACING_V2=true every LangChain / LangGraph call is
    automatically submitted to the configured LangSmith project as a traced run.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = str(config.langchain_tracing_v2).lower()
    os.environ["LANGCHAIN_API_KEY"] = config.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = config.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = config.langchain_endpoint
    logger.info(
        "LangSmith tracing %s — project: %s",
        "enabled" if config.langchain_tracing_v2 else "disabled",
        config.langchain_project,
    )


def _require(var: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise EnvironmentError(f"Required environment variable '{var}' is not set.")
    return value
