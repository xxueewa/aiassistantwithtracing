import chromadb
from langchain_core.documents import Document
from services.vector_store_local import create_vector_store, upsert_documents

# Add documents
vs = create_vector_store()
upsert_documents(vs, [
    Document(page_content="Liam is the assistant intelligence, she is good at searching and summarizing useful information and provides executable plans for the tasks She is built upon openai's LLM.", metadata={"source": "terminal"}),
    Document(page_content="Kat is the operator of this company, she has been here for about 1.5 years. She is graduated from Carnegie Mellon University.", metadata={"source": "terminal"}),
])

print("Documents inserted successfully.")