"""
rag_chain.py
------------
LangChain RAG chain: retrieve relevant chunks from OpenSearch, then
generate an answer with an OpenAI chat model.

LangSmith tracing is enabled via the @traceable decorator on run_rag_chain.
Because LANGCHAIN_TRACING_V2=true is set in the environment, every internal
LangChain call (retriever, LLM, prompt) is also automatically traced as a
child run inside the same LangSmith trace tree.
"""

import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langsmith import traceable

from src.config.config import create_config
from src.langchainapp.services.vector_store_local import create_vector_store

logger = logging.getLogger(__name__)
config_instance = create_config()

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the retrieved context below to answer "
    "the user's question as accurately as possible.\n"
    "If the context does not contain enough information, say so clearly "
    "rather than making something up.\n\n"
    "Context:\n{context}"
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


@traceable(name="rag-query", tags=["rag", "openai"])
def run_rag_chain(question: str) -> dict[str, Any]:
    """
    Full RAG pipeline:
      1. Embed the question and search for top-k chunks.
      2. Build a prompt with the retrieved context.
      3. Call the OpenAI chat model for the final answer.

    The @traceable decorator records this function as a top-level LangSmith
    run. All LangChain sub-calls (retriever, LLM, prompt) appear as children
    in the trace tree, giving end-to-end observability in LangSmith.
    """

    vs = create_vector_store(embedding_model=config_instance.embedding_model)
    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    llm = ChatOpenAI(model=config_instance.chat_model, temperature=0)

    # Build a LCEL chain:
    #   question → (retrieve context | passthrough question) → prompt → LLM → parse
    rag_chain = (
        RunnableParallel(
            {
                "context": retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)
    source_docs = retriever.invoke(question)

    return {
        "answer": answer,
        "source_documents": [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in source_docs
        ],
    }
