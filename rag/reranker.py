import re
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

RERANK_PROMPT = """Rate the relevance of this movie to the user's query.
Respond with ONLY a single integer from 0 to 10. No explanation.

Query: {query}
Movie: {title} ({year}) | Genres: {genres}
Description: {content}

Relevance score (0-10):"""

def llm_rerank(query: str, docs: list[Document], llm: OllamaLLM, top_k: int = 5) -> list[Document]:
    scored = []
    for doc in docs:
        prompt = RERANK_PROMPT.format(
            query=query,
            title=doc.metadata.get("title", "Unknown"),
            year=doc.metadata.get("year", "?"),
            genres=doc.metadata.get("genres", ""),
            content=doc.page_content[:400]
        )
        response = llm.invoke(prompt).strip()
        match = re.search(r"\b(10|[0-9])\b", response)
        score = float(match.group(1)) if match else 0.0
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
