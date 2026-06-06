import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # swap to "cuda" if you have a GPU
    )

def build_vectorstore(docs) -> Chroma:
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="movies"
    )
    print(f"Ingested {len(docs)} chunks into ChromaDB.")
    return vectorstore

def load_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
        collection_name="movies"
    )
