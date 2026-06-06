from fastapi import FastAPI
from pydantic import BaseModel
from mcp.tools import TOOLS
from pipeline.ingestor import load_vectorstore
from rag.retriever import get_retriever
from rag.chain import build_rag_chain

app = FastAPI(title="Movie RAG MCP Server")

vs        = load_vectorstore()
retriever = get_retriever(vs)
chain     = build_rag_chain(retriever)

@app.get("/tools")
def list_tools():
    return {"tools": TOOLS}

class CallRequest(BaseModel):
    name: str
    arguments: dict

@app.post("/call")
def call_tool(req: CallRequest):
    if req.name != "get_movie_recommendations":
        return {"error": f"Unknown tool: {req.name}"}

    query = req.arguments.get("query", "")
    k     = req.arguments.get("k", 5)

    retriever_k = get_retriever(vs, k=k)
    chain_k     = build_rag_chain(retriever_k, final_k=k)
    result      = chain_k.invoke(query)

    return {"tool": req.name, "result": result}

@app.get("/health")
def health():
    return {"status": "ok"}
