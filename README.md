# AI Movie Recommender

A RAG-based movie recommendation system powered by local LLMs via Ollama. It embeds the TMDB 5000 movies dataset into a ChromaDB vector store, exposes an MCP-compatible FastAPI server for semantic search and recommendations, and includes a LangChain ReAct agent for conversational interaction.

## Architecture

```
data/raw/tmdb_5000_movies.csv
        │
        ▼
  pipeline/          ← load, embed, ingest into ChromaDB
        │
        ▼
  chroma_db/         ← persistent vector store
        │
        ▼
  mcp/server.py      ← FastAPI MCP server (GET /tools, POST /call, GET /health)
        │
        ▼
  agent/agent.py     ← LangChain ReAct agent (conversational CLI)
```

**RAG pipeline:** Retrieves 15 candidate movies by vector similarity, deduplicates by title, LLM re-ranks to the top 5, then generates a natural-language recommendation with reasoning.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally with the `mistral` model pulled:
  ```bash
  ollama pull mistral
  ```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ingest the dataset into ChromaDB (only needed once)
python main.py
```

## Usage

### Start the MCP server

```bash
uvicorn mcp.server:app --reload
```

Server runs at `http://localhost:8000`.

| Endpoint | Description |
|---|---|
| `GET /tools` | List available tools |
| `POST /call` | Call a tool by name |
| `GET /health` | Health check |

### Run the conversational agent

In a separate terminal (with the server running):

```bash
python agent/agent.py
```

Example interaction:

```
You: Recommend me a sci-fi thriller with time travel
Agent: Based on your request, here are my top picks...
```

### Direct API call

```bash
curl -X POST http://localhost:8000/call \
  -H "Content-Type: application/json" \
  -d '{"name": "get_movie_recommendations", "arguments": {"query": "action movies set in space", "k": 5}}'
```

## Project Structure

```
movie-recommender/
├── data/raw/               # TMDB 5000 movies dataset
├── pipeline/
│   ├── loader.py           # CSV → DataFrame
│   ├── embedder.py         # DataFrame → LangChain Documents
│   └── ingestor.py         # Documents → ChromaDB
├── rag/
│   ├── retriever.py        # Vector similarity retriever
│   ├── reranker.py         # LLM-based re-ranker
│   └── chain.py            # Full RAG chain
├── mcp/
│   ├── server.py           # FastAPI MCP server
│   └── tools.py            # Tool definitions
├── agent/
│   └── agent.py            # ReAct agent CLI
├── chroma_db/              # Persisted vector store
├── main.py                 # Ingest script
└── requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

| Variable | Default | Description |
|---|---|---|
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Where ChromaDB stores data |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
