"""
scripts/test_api.py
-------------------
Smoke-test the deployed (or locally running) API.

Usage:
  # Against local SAM:
  API_URL=http://localhost:3000 python scripts/test_api.py
"""

import json
import os
import sys

import requests

BASE_URL = os.environ.get("API_URL", "http://localhost:3000").rstrip("/")


def test_ingest() -> None:
    print("── Ingest ──────────────────────────────────────")
    payload = {
        "texts": [
            (
                "LangSmith is a platform for LLM application development, "
                "monitoring, and testing built by LangChain."
            ),
            (
                "LangChain is an open-source framework that simplifies building "
                "applications powered by large language models."
            ),
            (
                "Amazon OpenSearch Service is a managed service that makes it easy "
                "to deploy, operate, and scale OpenSearch clusters in AWS."
            ),
        ],
        "metadata": [
            {"source": "langsmith-docs"},
            {"source": "langchain-docs"},
            {"source": "aws-docs"},
        ],
    }
    resp = requests.post(f"{BASE_URL}/ingest", json=payload, timeout=60)
    print(f"Status : {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))
    resp.raise_for_status()


def test_query(question: str = "What is LangSmith used for?") -> None:
    print("\n── Query ───────────────────────────────────────")
    print(f"Question: {question}")
    resp = requests.post(f"{BASE_URL}/query", json={"question": question}, timeout=60)
    print(f"Status : {resp.status_code}")
    body = resp.json()
    print(f"Answer :\n{body.get('answer', '')}")
    print(f"\nSources ({len(body.get('source_documents', []))}):")
    for doc in body.get("source_documents", []):
        print(f"  [{doc['metadata']}] {doc['content'][:80]}...")
    resp.raise_for_status()


if __name__ == "__main__":
    question_arg = " ".join(sys.argv[1:]) or "What is LangSmith used for?"
    test_ingest()
    test_query(question_arg)
