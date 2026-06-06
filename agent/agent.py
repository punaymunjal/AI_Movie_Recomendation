import requests
from langchain.tools import Tool
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

MCP_BASE = "http://localhost:8000"

def call_mcp_tool(query: str) -> str:
    response = requests.post(
        f"{MCP_BASE}/call",
        json={
            "name": "get_movie_recommendations",
            "arguments": {"query": query, "k": 5}
        }
    )
    return response.json().get("result", "No result returned.")

def build_agent():
    # dynamically discover tools from the MCP server
    tools_meta = requests.get(f"{MCP_BASE}/tools").json()["tools"]

    tools = [
        Tool(
            name=t["name"],
            description=t["description"],
            func=call_mcp_tool
        )
        for t in tools_meta
    ]

    llm    = ChatOllama(model="mistral", temperature=0)
    prompt = hub.pull("hwchase17/react")  # free community ReAct prompt

    agent    = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    return executor

if __name__ == "__main__":
    executor = build_agent()
    print("Movie agent ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        response = executor.invoke({"input": user_input})
        print(f"\nAgent: {response['output']}\n")
