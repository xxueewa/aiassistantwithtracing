import os
import chromadb

# 1. Connect to your persistent database directory
# Change path to match your actual database folder (e.g., './chroma_db' or './my_db')
DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_collection"

if not os.path.exists(DB_PATH):
    print(f"Error: Database directory '{DB_PATH}' not found.")
    exit(1)

client = chromadb.PersistentClient(path=DB_PATH)

try:
    # 2. Fetch the target collection
    collection = client.get_collection(name=COLLECTION_NAME)

    # 3. Get total document count
    total_docs = collection.count()
    print(f"--- Collection: '{COLLECTION_NAME}' ---")
    print(f"Total documents stored: {total_docs}\n")

    if total_docs == 0:
        print("The collection is empty.")
    else:
        # 4. Retrieve all documents, IDs, and metadata
        # By default, collection.get() fetches all elements if no IDs are passed
        results = collection.get(
            include=["documents", "metadatas", "embeddings"]  # Omit "embeddings" if you don't want vectors printed
        )

        # 5. Iterate and print out the items cleanly
        for i in range(total_docs):
            doc_id = results["ids"][i]
            # Use .get() to avoid KeyErrors if some elements were not saved
            doc_content = results["documents"][i] if results["documents"] else "None"
            metadata = results["metadatas"][i] if results["metadatas"] else {}

            print(f"[{i + 1}] ID: {doc_id}")
            print(f"    Metadata: {metadata}")
            print(f"    Content:  {doc_content}")
            print("-" * 50)

except ValueError:
    print(f"Error: Collection '{COLLECTION_NAME}' does not exist in this database.")
    # Optional: List available collection names to help the user debug
    print("Available collections:", client.list_collections())
