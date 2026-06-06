import os
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

def get_retriever(vectorstore: Chroma, k: int = 5) -> VectorStoreRetriever:
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )