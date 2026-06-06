from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from rag.reranker import llm_rerank

PROMPT_TEMPLATE = """
You are an expert movie critic and recommendation engine.
Using ONLY the movies listed in the context below, recommend
the top 3 to 5 best matches for the user's request.
For each movie, explain specifically why it fits, referencing
its genre, plot, or other details from the context.
Do not recommend movies not in the context.

Context:
{context}

User request: {question}

Recommendations (list 3-5 movies):
"""

def format_docs(docs):
    return "\n\n".join(
        f"- {d.metadata['title']} ({d.metadata['year']}) "
        f"[{d.metadata['genres']}]: {d.page_content}"
        for d in docs
    )

def build_rag_chain(retriever, fetch_k: int = 15, final_k: int = 5):
    rerank_llm = OllamaLLM(model="mistral", temperature=0)
    gen_llm    = OllamaLLM(model="mistral", temperature=0.3)
    prompt     = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    wide_retriever = retriever.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": fetch_k}
    )

    def retrieve_and_rerank(query: str) -> str:
        docs = wide_retriever.invoke(query)
        # deduplicate chunks from the same movie before re-ranking
        seen, unique_docs = set(), []
        for doc in docs:
            title = doc.metadata.get("title")
            if title not in seen:
                seen.add(title)
                unique_docs.append(doc)
        reranked = llm_rerank(query, unique_docs, rerank_llm, top_k=final_k)
        return format_docs(reranked)

    chain = (
        {"context": RunnableLambda(retrieve_and_rerank), "question": RunnablePassthrough()}
        | prompt
        | gen_llm
        | StrOutputParser()
    )
    return chain
