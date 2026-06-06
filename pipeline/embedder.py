from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def to_documents(movies: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = []
    for movie in movies:
        chunks = splitter.create_documents(
            texts=[movie["text"]],
            metadatas=[movie["metadata"]]
        )
        docs.extend(chunks)
    return docs