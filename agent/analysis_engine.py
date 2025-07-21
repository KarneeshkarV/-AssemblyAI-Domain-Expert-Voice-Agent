import asyncio
import os
from typing import Optional

from agno.agent import Agent
from agno.embedder.ollama import OllamaEmbedder
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase, PDFUrlReader
from agno.knowledge.text import TextKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.team.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools
from agno.vectordb.qdrant import Qdrant

api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
collection_name = "Default"

vector_db = Qdrant(
    collection=collection_name,
    url=os.getenv("QDRANT_URL") or "http://localhost:6333",
    embedder=OllamaEmbedder(id="nomic-embed-text:latest", dimensions=768),
)

knowledge_base = PDFUrlKnowledgeBase(
    urls=[
        "https://www.fidelity.com/bin-public/060_www_fidelity_com/documents/wealth-planning_investment-strategy.pdf"
    ],
    vector_db=vector_db,
    num_documents=5,
)
# asyncio.run(knowledge_base.aload(recreate=True))


def finance_agent(message: str, user: str = "user"):
    run_id: Optional[str] = None
    rag_agent = Agent(
        knowledge=knowledge_base,
        name="Personal Knowledge Agent",
        role="Handles personal knowledge and gives out the most relevant information",
        search_knowledge=True,
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
        members=[postive_web_agent, rag_agent, finance_agent ,negative_web_agent],
        tools=[ReasoningTools(add_instructions=True)],
        instructions=[
            "Collaborate to provide comprehensive financial and investment insights",
            "You will be given both positive and negative information about the topic , be unbiased and provide only the most relevant information and which will be good for the user's long term goal",
            "Consider both fundamental analysis and market sentiment",
            "Use tables and charts to display data clearly and professionally",
            "Present findings in a structured, easy-to-follow format",
            "Only output the final consolidated analysis, not individual agent responses",
        ],
        markdown=True,
        show_members_responses=True,
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
