TOOLS = [
    {
        "name": "get_movie_recommendations",
        "description": (
            "Retrieves personalized movie recommendations based on a "
            "natural language query. Uses semantic search over a movie "
            "database to find relevant films and generates context-aware "
            "suggestions with explanations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of the desired movie"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of candidates to retrieve (default 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
]
