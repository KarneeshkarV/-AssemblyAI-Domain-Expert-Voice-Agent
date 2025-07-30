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
from agno.vectordb.qdrant import Qdrant
from rich.pretty import pprint

from qdrant_rag.rag_client import RagClient

api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
legal_collection_name = "Legal"

user_id = "Legal User"
legal_vector_db = Qdrant(
    collection=legal_collection_name,
    url=os.getenv("QDRANT_URL") or "http://localhost:6333",
    embedder=OllamaEmbedder(id="nomic-embed-text:latest", dimensions=768),
)

# Separate database file for legal data
legal_db_file = "tmp/legal_agent.db"

legal_memory = Memory(
    model=OpenAIChat(id="gpt-4.1"),
    db=SqliteMemoryDb(table_name="legal_user_memories", db_file=legal_db_file),
)

legal_storage = SqliteStorage(table_name="legal_agent_sessions", db_file=legal_db_file)

# Legal knowledge base with relevant legal documents and resources
legal_knowledge_base = PDFUrlKnowledgeBase(
    urls=[
        "https://www.americanbar.org/content/dam/aba/administrative/professional_responsibility/model_rules_2020.pdf",  # ABA Model Rules
        "https://www.law.cornell.edu/constitution/constitution.overview.html",  # Constitutional Law Reference
    ],
    vector_db=legal_vector_db,
    num_documents=10,
)


def legal_logger_hook(function_name: str, function_call: Callable, arguments: Dict[str, Any]):
    """Hook function that wraps legal tool execution"""
    print(f"[LEGAL] About to call {function_name} with arguments: {arguments}")
    result = function_call(**arguments)
    print(f"[LEGAL] Function call completed with result: {result}")
    return result


@tool(
    name="get_legal_rag_data",
    description="Tool to retrieve legal knowledge from RAG system including case law, statutes, and legal precedents",
    show_result=True,
    stop_after_tool_call=True,
    tool_hooks=[legal_logger_hook],
    requires_confirmation=True,
    cache_results=True,
    cache_dir="/tmp/legal_agno_cache",
    cache_ttl=3600,
)
def legal_rag_tool_query(message: str, collection_name: str = "Legal", limit: int = 5):
    """Query legal knowledge base for relevant legal information, precedents, and statutes"""
    rag_client = RagClient(collection_name=collection_name)
    return rag_client.retrieve_documents(message)


@tool(
    name="legal_disclaimer_tool",
    description="Provides essential legal disclaimers and attorney-client privilege notices",
    show_result=True,
)
def legal_disclaimer():
    """Returns standard legal disclaimer and attorney-client privilege notice"""
    return """
    ⚖️ LEGAL DISCLAIMER:
    This information is for educational and informational purposes only and does not constitute legal advice.
    No attorney-client relationship is created through this interaction.
    Legal advice must be tailored to the specific circumstances of each case.
    Always consult with qualified legal counsel for specific legal matters.
    Laws vary by jurisdiction and change frequently.
    This analysis should not be relied upon for legal decision-making.
    """


@tool(
    name="jurisdiction_warning_tool",
    description="Provides jurisdiction-specific warnings and legal practice notices",
    show_result=True,
)
def jurisdiction_warning():
    """Returns jurisdiction-specific warnings for legal analysis"""
    return """
    📍 JURISDICTION NOTICE:
    Legal requirements, procedures, and interpretations vary significantly by jurisdiction.
    This analysis may not reflect the laws of your specific location.
    Federal, state, and local laws may all apply to your situation.
    Always verify current law in your relevant jurisdiction(s).
    Consult local legal counsel familiar with applicable jurisdictional requirements.
    """


def legal_memory_agent_query(message: str, debug: bool = True, tui: bool = True):
    """Query legal memory agent for legal history and context"""
    legal_memory_agent = Agent(
        model=OpenAIChat(id="gpt-4.1"),
        memory=legal_memory,
        enable_agentic_memory=True,
        enable_user_memories=True,
        storage=legal_storage,
        add_history_to_messages=True,
        num_history_runs=3,
        markdown=True,
        debug_mode=debug,
        instructions=[
            "You are a legal memory agent specializing in legal case history and context",
            "Always maintain attorney-client privilege and legal confidentiality standards",
            "Provide relevant legal precedents and case history when queried",
            "Include appropriate legal disclaimers in all responses",
        ],
    )

    pprint(legal_memory.get_user_memories(user_id=user_id))
    
    if tui:
        legal_memory_agent.print_response(message, user_id=user_id)
    else:
        response = legal_memory_agent.run(message, user_id=user_id)
        print(response.content)


def create_legal_team(user: str = "legal_user", debug: bool = False) -> Team:
    """Create and return a persistent legal team instance for conversation."""
    # Legal Research Agent
    legal_research_agent = Agent(
        knowledge=legal_knowledge_base,
        name="Legal Research Agent",
        role="Legal research specialist focusing on case law, statutes, regulations, and legal precedents",
        search_knowledge=True,
        tools=[legal_rag_tool_query, DuckDuckGoTools()],
        model=OpenAIChat("gpt-4.1"),
        instructions=[
            "Focus on comprehensive legal research and case law analysis",
            "Provide accurate citations and legal precedents",
            "Consider both federal and state law implications",
            "Always include source attribution for legal authorities",
        ],
        show_tool_calls=True,
        debug_mode=debug,
    )
    
    # Contract Analysis Agent
    contract_analysis_agent = Agent(
        name="Contract Analysis Agent",
        role="Contract review and analysis specialist focusing on terms, conditions, and legal implications",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[legal_rag_tool_query, DuckDuckGoTools()],
        instructions=[
            "Analyze contract terms, conditions, and legal implications",
            "Identify potential risks, ambiguities, and missing provisions",
            "Consider enforceability and compliance issues",
            "Provide recommendations for contract improvements",
        ],
        add_datetime_to_instructions=True,
        debug_mode=debug,
    )
    
    # Regulatory Compliance Agent
    regulatory_compliance_agent = Agent(
        name="Regulatory Compliance Agent",
        role="Regulatory compliance specialist covering industry regulations and legal requirements",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[DuckDuckGoTools(), legal_rag_tool_query],
        instructions=[
            "Analyze regulatory compliance requirements and obligations",
            "Consider industry-specific regulations and standards",
            "Identify compliance gaps and remediation strategies",
            "Stay current with evolving regulatory landscapes",
        ],
        add_datetime_to_instructions=True,
        debug_mode=debug,
    )
    
    # Risk Assessment Agent
    risk_assessment_agent = Agent(
        name="Legal Risk Assessment Agent",
        role="Legal risk analysis specialist focusing on liability assessment and risk mitigation",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[DuckDuckGoTools(), legal_rag_tool_query],
        instructions=[
            "Assess legal risks and potential liability exposure",
            "Analyze risk mitigation strategies and legal protections",
            "Consider litigation risks and dispute resolution options",
            "Evaluate insurance and indemnification considerations",
        ],
        add_datetime_to_instructions=True,
        debug_mode=debug,
    )
    
    # Legal Ethics Agent
    legal_ethics_agent = Agent(
        name="Legal Ethics Agent",
        role="Professional responsibility and legal ethics specialist",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[legal_disclaimer, jurisdiction_warning, legal_rag_tool_query],
        instructions=[
            "Ensure adherence to professional responsibility standards",
            "Provide ethical guidance for legal practitioners",
            "Consider attorney-client privilege and confidentiality requirements",
            "Include appropriate legal disclaimers and ethical notices",
        ],
        add_datetime_to_instructions=True,
        debug_mode=debug,
    )

    # Legal Reasoning Team
    legal_reasoning_team = Team(
        user_id=user,
        name="Legal Analysis Team",
        mode="coordinate",
        model=OpenAIChat(id="o4-mini-2025-04-16"),
        members=[
            legal_research_agent,
            contract_analysis_agent,
            regulatory_compliance_agent,
            risk_assessment_agent,
            legal_ethics_agent,
        ],
        tools=[ReasoningTools(add_instructions=True), legal_disclaimer, jurisdiction_warning],
        instructions=[
            "Collaborate to provide comprehensive legal analysis and guidance",
            "Always prioritize accuracy, ethical compliance, and professional responsibility",
            "Consider multiple legal perspectives and jurisdictional variations",
            "Include appropriate legal disclaimers and attorney-client privilege notices",
            "Use structured legal reasoning and analysis methodologies",
            "Reference authoritative legal sources and current law",
            "Emphasize the importance of consulting qualified legal counsel",
            "Present findings in a clear, legally accurate format",
            "Always include legal disclaimers and jurisdiction warnings in final output",
        ],
        markdown=True,
        show_members_responses=True,
        memory=legal_memory,
        enable_agentic_memory=True,
        enable_user_memories=True,
        storage=legal_storage,
        add_history_to_messages=True,
        num_history_runs=3,
        enable_agentic_context=True,
        add_datetime_to_instructions=True,
        debug_mode=debug,
        success_criteria="The legal team has provided a comprehensive, accurate legal analysis with appropriate disclaimers, jurisdictional considerations, and clear recommendations while emphasizing the need for qualified legal counsel.",
    )
    return legal_reasoning_team


def legal_analysis_team(message: str, user: str = "legal_user", debug: bool = True, tui: bool = True):
    """Main legal analysis team function with specialized legal agents"""
    run_id: Optional[str] = None
    
    # Legal Research Agent
    legal_research_agent = Agent(
        knowledge=legal_knowledge_base,
        name="Legal Research Agent",
        role="Legal research specialist focusing on case law, statutes, regulations, and legal precedents",
        search_knowledge=True,
        tools=[legal_rag_tool_query, DuckDuckGoTools()],
        model=OpenAIChat("gpt-4.1"),
        instructions=[
            "Focus on comprehensive legal research and case law analysis",
            "Provide accurate citations and legal precedents",
            "Consider both federal and state law implications",
            "Always include source attribution for legal authorities",
        ],
        show_tool_calls=True,
        debug_mode=debug,
    )
    
    # Contract Analysis Agent
    contract_analysis_agent = Agent(
        name="Contract Analysis Agent",
        role="Contract review and analysis specialist focusing on terms, conditions, and legal implications",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[legal_rag_tool_query, DuckDuckGoTools()],
        instructions=[
            "Analyze contract terms, conditions, and legal implications",
            "Identify potential risks, ambiguities, and missing provisions",
            "Consider enforceability and compliance issues",
            "Provide recommendations for contract improvements",
        ],
        add_datetime_to_instructions=True,
        debug_mode=debug,
    )
    
    # Regulatory Compliance Agent
    regulatory_compliance_agent = Agent(
        name="Regulatory Compliance Agent",
        role="Regulatory compliance specialist covering industry regulations and legal requirements",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[DuckDuckGoTools(), legal_rag_tool_query],
        instructions=[
            "Analyze regulatory compliance requirements and obligations",
            "Consider industry-specific regulations and standards",
            "Identify compliance gaps and remediation strategies",
            "Stay current with evolving regulatory landscapes",
        ],
        add_datetime_to_instructions=True,
        debug_mode=debug,
    )
    
    # Risk Assessment Agent
    risk_assessment_agent = Agent(
        name="Legal Risk Assessment Agent",
        role="Legal risk analysis specialist focusing on liability assessment and risk mitigation",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[DuckDuckGoTools(), legal_rag_tool_query],
        instructions=[
            "Assess legal risks and potential liability exposure",
            "Analyze risk mitigation strategies and legal protections",
            "Consider litigation risks and dispute resolution options",
            "Evaluate insurance and indemnification considerations",
        ],
        add_datetime_to_instructions=True,
        debug_mode=debug,
    )
    
    # Legal Ethics Agent
    legal_ethics_agent = Agent(
        name="Legal Ethics Agent",
        role="Professional responsibility and legal ethics specialist",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[legal_disclaimer, jurisdiction_warning, legal_rag_tool_query],
        instructions=[
            "Ensure adherence to professional responsibility standards",
            "Provide ethical guidance for legal practitioners",
            "Consider attorney-client privilege and confidentiality requirements",
            "Include appropriate legal disclaimers and ethical notices",
        ],
        add_datetime_to_instructions=True,
        debug_mode=debug,
    )

    # Legal Reasoning Team
    legal_reasoning_team = Team(
        user_id=user,
        name="Legal Analysis Team",
        mode="coordinate",
        model=OpenAIChat(id="o4-mini-2025-04-16"),
        members=[
            legal_research_agent,
            contract_analysis_agent,
            regulatory_compliance_agent,
            risk_assessment_agent,
            legal_ethics_agent,
        ],
        tools=[ReasoningTools(add_instructions=True), legal_disclaimer, jurisdiction_warning],
        instructions=[
            "Collaborate to provide comprehensive legal analysis and guidance",
            "Always prioritize accuracy, ethical compliance, and professional responsibility",
            "Consider multiple legal perspectives and jurisdictional variations",
            "Include appropriate legal disclaimers and attorney-client privilege notices",
            "Use structured legal reasoning and analysis methodologies",
            "Reference authoritative legal sources and current law",
            "Emphasize the importance of consulting qualified legal counsel",
            "Present findings in a clear, legally accurate format",
            "Always include legal disclaimers and jurisdiction warnings in final output",
        ],
        markdown=True,
        show_members_responses=True,
        memory=legal_memory,
        enable_agentic_memory=True,
        enable_user_memories=True,
        storage=legal_storage,
        add_history_to_messages=True,
        num_history_runs=3,
        enable_agentic_context=True,
        add_datetime_to_instructions=True,
        debug_mode=debug,
        success_criteria="The legal team has provided a comprehensive, accurate legal analysis with appropriate disclaimers, jurisdictional considerations, and clear recommendations while emphasizing the need for qualified legal counsel.",
    )
    
    if run_id is None:
        run_id = legal_reasoning_team.run_id
        print(f"Started Legal Analysis Run: {run_id}\n")
    else:
        print(f"Continuing Legal Analysis Run: {run_id}\n")

    if tui:
        legal_reasoning_team.print_response(message)
    else:
        response = legal_reasoning_team.run(message)
        print(response.content)