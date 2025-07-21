import os
from typing import Any, Callable, Dict, Optional

from agno.agent import Agent
from agno.embedder.ollama import OllamaEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from agno.models.openai import OpenAIChat
from agno.storage.sqlite import SqliteStorage
from agno.team.team import Team
from agno.tools import tool
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools
from agno.vectordb.qdrant import Qdrant
from rich.pretty import pprint

from qdrant_rag.rag_client import RagClient

api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
collection_name = "Default"

user_id = "Test User"
vector_db = Qdrant(
    collection=collection_name,
    url=os.getenv("QDRANT_URL") or "http://localhost:6333",
    embedder=OllamaEmbedder(id="nomic-embed-text:latest", dimensions=768),
)
# Database file for memory and storage
db_file = "tmp/agent.db"

memory = Memory(
    # Use any model for creating memories
    model=OpenAIChat(id="gpt-4.1"),
    db=SqliteMemoryDb(table_name="user_memories", db_file=db_file),
)

storage = SqliteStorage(table_name="agent_sessions", db_file=db_file)
knowledge_base = PDFUrlKnowledgeBase(
    urls=[
        "https://www.fidelity.com/bin-public/060_www_fidelity_com/documents/wealth-planning_investment-strategy.pdf"
    ],
    vector_db=vector_db,
    num_documents=5,
)


# asyncio.run(knowledge_base.aload(recreate=True))
def logger_hook(function_name: str, function_call: Callable, arguments: Dict[str, Any]):
    """Hook function that wraps the tool execution"""
    print(f"About to call {function_name} with arguments: {arguments}")
    result = function_call(**arguments)
    print(f"Function call completed with result: {result}")
    return result


@tool(
    name="get_data_from_rag",
    description="Tool to get data from RAG",
    show_result=True,
    stop_after_tool_call=True,
    tool_hooks=[logger_hook],
    requires_confirmation=True,
    cache_results=True,
    cache_dir="/tmp/agno_cache",
    cache_ttl=3600,
)
def rag_tool_query(message: str, collection_name: str = "Default", limit: int = 3):
    rag_client = RagClient(collection_name=collection_name)
    return rag_client.retrieve_documents(message)


memory_agent = Agent(
    model=OpenAIChat(id="gpt-4.1"),
    memory=memory,
    enable_agentic_memory=True,
    enable_user_memories=True,
    storage=storage,
    add_history_to_messages=True,
    num_history_runs=3,
    markdown=True,
)


def memory_agent_query(message: str):
    memory_agent = Agent(
        model=OpenAIChat(id="gpt-4.1"),
        memory=memory,
        enable_agentic_memory=True,
        enable_user_memories=True,
        storage=storage,
        add_history_to_messages=True,
        num_history_runs=3,
        markdown=True,
    )

    pprint(memory.get_user_memories(user_id=user_id))
    memory_agent.print_response(message, user_id=user_id)


def finance_agent(message: str, user: str = "user"):
    run_id: Optional[str] = None
    rag_agent = Agent(
        knowledge=knowledge_base,
        name="Personal Knowledge Agent",
        role="Handles personal knowledge and gives out the most relevant information",
        search_knowledge=True,
        tools=[rag_tool_query],
        model=OpenAIChat("gpt-4.1"),
        show_tool_calls=True,
        debug_mode=True,
    )
    postive_web_agent = Agent(
        name="Web Search Agent",
        role="Handle web search requests and general research and gives only the positive results about the topic",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[DuckDuckGoTools()],
        instructions="Always include sources",
        add_datetime_to_instructions=True,
    )
    negative_web_agent = Agent(
        name="Web Search Agent",
        role="Handle web search requests and general research and gives only the nrgative results about the topic",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[DuckDuckGoTools()],
        instructions="Always include sources",
        add_datetime_to_instructions=True,
    )

    finance_agent = Agent(
        name="Finance Agent",
        role="Handle financial data requests and market analysis",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[
            YFinanceTools(
                stock_price=True,
                stock_fundamentals=True,
                analyst_recommendations=True,
                company_info=True,
            )
        ],
        instructions=[
            "Use tables to display stock prices, fundamentals (P/E, Market Cap), and recommendations.",
            "Clearly state the company name and ticker symbol.",
            "Focus on delivering actionable financial insights.",
        ],
        add_datetime_to_instructions=True,
    )
    reasoning_finance_team = Team(
        user_id=user,
        name="Reasoning Finance Team",
        mode="coordinate",
        model=OpenAIChat(id="o4-mini-2025-04-16"),
        members=[
            postive_web_agent,
            rag_agent,
            finance_agent,
            negative_web_agent,
        ],
        tools=[ReasoningTools(add_instructions=True)],
        instructions=[
            "Always use memory agent to query about the previous user's memories and the topic and store the current user's memories about the topic",
            "Collaborate to provide comprehensive financial and investment insights",
            "You will be given both positive and negative information about the topic , be unbiased and provide only the most relevant information and which will be good for the user's long term goal",
            "Consider both fundamental analysis and market sentiment",
            "Use tables and charts to display data clearly and professionally",
            "Present findings in a structured, easy-to-follow format",
            "Only output the final consolidated analysis, not individual agent responses",
        ],
        markdown=True,
        show_members_responses=True,
        memory=memory,
        enable_agentic_memory=True,
        enable_user_memories=True,
        storage=storage,
        add_history_to_messages=True,
        num_history_runs=3,
        enable_agentic_context=True,
        add_datetime_to_instructions=True,
        success_criteria="The team has provided a complete financial analysis with data, visualizations, risk assessment, and actionable investment recommendations supported by quantitative analysis and market research.",
    )
    if run_id is None:
        run_id = reasoning_finance_team.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    reasoning_finance_team.print_response(message)
