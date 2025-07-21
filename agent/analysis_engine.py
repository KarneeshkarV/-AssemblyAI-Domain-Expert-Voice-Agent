import os
from typing import Optional

import typer
from agno.agent import Agent, AgentKnowledge
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools
from agno.vectordb.qdrant import Qdrant

api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
collection_name = "Default"

vector_db = Qdrant(
    collection=collection_name,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

knowledge_base = AgentKnowledge(
    vector_db=vector_db,

)


knowledge_base.load(recreate=True, upsert=True)
def qdrant_agent(message: str, user: str = "user"):
    run_id: Optional[str] = None

    agent = Agent(
        user_id=user,
        knowledge=knowledge_base,
        search_knowledge=True,
        model=OpenAIChat("gpt-4o"),
        tools=[YFinanceTools(stock_price=True)],
        show_tool_calls=True,
        debug_mode=True,
    )

    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    agent.print_response(message)


if __name__ == "__main__":
    # Comment out after first run

    typer.run(qdrant_agent)
