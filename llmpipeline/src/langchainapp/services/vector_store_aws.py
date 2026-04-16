# """
# vector_store_aws.py
# ---------------
# Vector store backed by Amazon OpenSearch Service (serverless-friendly).
#
# Authentication uses AWS SigV4 (IAM role of the Lambda execution role) — no
# username/password required. The LangChain OpenSearchVectorSearch integration
# handles both indexing and k-NN similarity search.
#
# Lambda warm-start:  the OpenSearchVectorSearch client is cached at module
# level so repeated invocations reuse the same HTTP connection pool.
#
# integrate later after dev completed
# """
#
# import logging
# from typing import Optional
#
# import boto3
# from opensearchpy import OpenSearch, RequestsHttpConnection
# from requests_aws4auth import AWS4Auth
# from langchain_community.vectorstores import OpenSearchVectorSearch
# from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings
#
# logger = logging.getLogger(__name__)
#
# # Warm-start cache
# _vs_cache: Optional[OpenSearchVectorSearch] = None
#
#
# # ── Embedding helper ──────────────────────────────────────────────────────────
#
# def get_embeddings(model: str) -> OpenAIEmbeddings:
#     return OpenAIEmbeddings(model=model)
#
#
# # ── AWS SigV4 auth ────────────────────────────────────────────────────────────
#
# def _get_aws_auth(region: str) -> AWS4Auth:
#     """Build AWS4Auth from the Lambda execution role credentials."""
#     credentials = boto3.Session().get_credentials().get_frozen_credentials()
#     return AWS4Auth(
#         credentials.access_key,
#         credentials.secret_key,
#         region,
#         "es",
#         session_token=credentials.token,
#     )
#
#
# # ── Client factory ────────────────────────────────────────────────────────────
#
# def get_vector_store(
#     endpoint: str,
#     index_name: str,
#     embeddings: OpenAIEmbeddings,
#     region: str,
#     *,
#     force_new: bool = False,
# ) -> OpenSearchVectorSearch:
#     """
#     Return a LangChain OpenSearchVectorSearch client connected to the given
#     OpenSearch Service domain endpoint.
#
#     The client is cached after the first call (warm-start optimisation).
#     Pass force_new=True to bypass the cache (e.g. after an index rebuild).
#     """
#     global _vs_cache
#     if _vs_cache is not None and not force_new:
#         logger.debug("Returning cached OpenSearch vector store.")
#         return _vs_cache
#
#     auth = _get_aws_auth(region)
#     vs = OpenSearchVectorSearch(
#         opensearch_url=f"https://{endpoint}",
#         index_name=index_name,
#         embedding_function=embeddings,
#         http_auth=auth,
#         use_ssl=True,
#         verify_certs=True,
#         connection_class=RequestsHttpConnection,
#         # k-NN engine used for approximate nearest-neighbour search
#         engine="nmslib",
#         space_type="cosinesimil",
#     )
#     _vs_cache = vs
#     logger.info("Connected to OpenSearch domain: %s  index: %s", endpoint, index_name)
#     return vs
#
#
# # ── Document ingestion ────────────────────────────────────────────────────────
#
# def upsert_documents(
#     documents: list[Document],
#     endpoint: str,
#     index_name: str,
#     embeddings: OpenAIEmbeddings,
#     region: str,
# ) -> OpenSearchVectorSearch:
#     """
#     Embed *documents* and add them to the OpenSearch index.
#     The index is created automatically on first write if it does not exist.
#     """
#     vs = get_vector_store(endpoint, index_name, embeddings, region, force_new=False)
#     vs.add_documents(documents)
#     logger.info("Upserted %d document(s) into index '%s'.", len(documents), index_name)
#     return vs
