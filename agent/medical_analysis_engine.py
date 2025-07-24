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
medical_collection_name = "Default"

user_id = "Medical User"
medical_vector_db = Qdrant(
    collection=medical_collection_name,
    url=os.getenv("QDRANT_URL") or "http://localhost:6333",
    embedder=OllamaEmbedder(id="nomic-embed-text:latest", dimensions=768),
)

# Separate database file for medical data
medical_db_file = "tmp/medical_agent.db"

medical_memory = Memory(
    model=OpenAIChat(id="gpt-4.1"),
    db=SqliteMemoryDb(table_name="medical_user_memories", db_file=medical_db_file),
)

medical_storage = SqliteStorage(table_name="medical_agent_sessions", db_file=medical_db_file)

# Medical knowledge base with relevant medical literature
medical_knowledge_base = PDFUrlKnowledgeBase(
    urls=[
        "https://www.who.int/publications/i/item/9789241549363",  # WHO Medical Device Regulations
        "https://www.fda.gov/files/drugs/published/Clinical-Pharmacology-and-Biopharmaceutics-Review.pdf",  # FDA Clinical Pharmacology
    ],
    vector_db=medical_vector_db,
    num_documents=10,
)


def medical_logger_hook(function_name: str, function_call: Callable, arguments: Dict[str, Any]):
    """Hook function that wraps medical tool execution"""
    print(f"[MEDICAL] About to call {function_name} with arguments: {arguments}")
    result = function_call(**arguments)
    print(f"[MEDICAL] Function call completed with result: {result}")
    return result


@tool(
    name="get_medical_rag_data",
    description="Tool to retrieve medical knowledge from RAG system",
    show_result=True,
    stop_after_tool_call=True,
    tool_hooks=[medical_logger_hook],
    requires_confirmation=True,
    cache_results=True,
    cache_dir="/tmp/medical_agno_cache",
    cache_ttl=3600,
)
def medical_rag_tool_query(message: str, collection_name: str = "Medical", limit: int = 5):
    """Query medical knowledge base for relevant medical information"""
    rag_client = RagClient(collection_name=collection_name)
    return rag_client.retrieve_documents(message)


@tool(
    name="medical_safety_check",
    description="Performs safety checks and adds medical disclaimers",
    show_result=True,
)
def medical_safety_disclaimer():
    """Returns standard medical safety disclaimer"""
    return """
    ⚠️ MEDICAL DISCLAIMER:
    This information is for educational purposes only and should not replace professional medical advice.
    Always consult with qualified healthcare professionals for medical decisions.
    In case of medical emergency, contact emergency services immediately.
    """


def medical_memory_agent_query(message: str, debug: bool = True, tui: bool = True):
    """Query medical memory agent for patient history and medical context"""
    medical_memory_agent = Agent(
        model=OpenAIChat(id="gpt-4.1"),
        memory=medical_memory,
        enable_agentic_memory=True,
        enable_user_memories=True,
        storage=medical_storage,
        add_history_to_messages=True,
        num_history_runs=3,
        markdown=True,
        debug_mode=debug,
        instructions=[
            "You are a medical memory agent specializing in patient history and medical context",
            "Always maintain patient confidentiality and follow medical ethics",
            "Provide relevant medical history when queried",
        ],
    )

    pprint(medical_memory.get_user_memories(user_id=user_id))
    
    if tui:
        medical_memory_agent.print_response(message, user_id=user_id)
    else:
        response = medical_memory_agent.run(message, user_id=user_id)
        print(response.content)


def create_medical_team(user: str = "medical_user", debug: bool = False) -> Team:
    """Create and return a persistent medical team instance for conversation."""
    # Clinical Diagnostic Agent
    clinical_diagnostic_agent = Agent(
        knowledge=medical_knowledge_base,
        name="Clinical Diagnostic Agent",
        role="Primary diagnostic reasoning specialist with expertise in differential diagnosis and clinical assessment",
        search_knowledge=True,
        tools=[medical_rag_tool_query, medical_safety_disclaimer],
        model=OpenAIChat("gpt-4.1"),
        instructions=[
            "Focus on evidence-based diagnostic reasoning",
            "Consider differential diagnoses systematically",
            "Always include clinical assessment guidelines",
            "Emphasize the importance of proper medical evaluation",
        ],
        show_tool_calls=True,
        debug_mode=debug,
    )
    
    # Medical Research Agent
    medical_research_agent = Agent(
        name="Medical Research Agent",
        role="Medical literature analysis specialist focused on evidence-based medicine and latest research findings",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[DuckDuckGoTools(), medical_rag_tool_query],
        instructions=[
            "Search for peer-reviewed medical literature",
            "Focus on evidence-based medicine principles",
            "Include publication dates and study quality",
            "Always cite reputable medical sources",
        ],
        add_datetime_to_instructions=True,
        debug_mode=debug,
    )
    
    # Pharmacology Agent
    pharmacology_agent = Agent(
        name="Pharmacology Agent",
        role="Drug interactions, dosing, and pharmaceutical safety specialist",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[DuckDuckGoTools(), medical_rag_tool_query],
        instructions=[
            "Analyze drug interactions and contraindications",
            "Provide dosing guidelines and safety information",
            "Consider pharmacokinetics and pharmacodynamics",
            "Emphasize adverse effects and monitoring requirements",
        ],
        add_datetime_to_instructions=True,
        debug_mode=debug,
    )
    
    # Specialist Consultation Agent
    specialist_consultation_agent = Agent(
        name="Specialist Consultation Agent",
        role="Multi-specialty medical expertise covering cardiology, neurology, endocrinology, and other specialties",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[DuckDuckGoTools(), medical_rag_tool_query],
        instructions=[
            "Provide specialty-specific medical insights",
            "Reference specialty guidelines and procedures",
            "Consider multidisciplinary approaches",
            "Highlight when specialist referral is needed",
        ],
        add_datetime_to_instructions=True,
        debug_mode=debug,
    )
    
    # Patient Safety Agent
    patient_safety_agent = Agent(
        name="Patient Safety Agent",
        role="Risk assessment and patient safety protocols specialist",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[medical_safety_disclaimer, medical_rag_tool_query],
        instructions=[
            "Assess patient safety risks and considerations",
            "Provide safety protocols and quality measures",
            "Identify potential safety concerns",
            "Always include appropriate medical disclaimers",
        ],
        add_datetime_to_instructions=True,
        debug_mode=debug,
    )

    # Medical Reasoning Team
    medical_reasoning_team = Team(
        user_id=user,
        name="Medical Analysis Team",
        mode="coordinate",
        model=OpenAIChat(id="o4-mini-2025-04-16"),
        members=[
            clinical_diagnostic_agent,
            medical_research_agent,
            pharmacology_agent,
            specialist_consultation_agent,
            patient_safety_agent,
        ],
        tools=[ReasoningTools(add_instructions=True), medical_safety_disclaimer],
        instructions=[
            "Collaborate to provide comprehensive medical analysis and insights",
            "Always prioritize patient safety and evidence-based medicine",
            "Consider multiple medical perspectives and specialties",
            "Include appropriate medical disclaimers and safety warnings",
            "Use structured medical reasoning and differential diagnosis approaches",
            "Reference peer-reviewed medical literature and clinical guidelines",
            "Emphasize the importance of professional medical consultation",
            "Present findings in a clear, medically accurate format",
            "Always include the medical safety disclaimer in final output",
        ],
        markdown=True,
        show_members_responses=True,
        memory=medical_memory,
        enable_agentic_memory=True,
        enable_user_memories=True,
        storage=medical_storage,
        add_history_to_messages=True,
        num_history_runs=3,
        enable_agentic_context=True,
        add_datetime_to_instructions=True,
        debug_mode=debug,
        success_criteria="The medical team has provided a comprehensive, evidence-based medical analysis with appropriate safety considerations, literature references, and clear recommendations while emphasizing the need for professional medical consultation.",
    )
    return medical_reasoning_team


def medical_analysis_team(message: str, user: str = "medical_user", debug: bool = True, tui: bool = True):
    """Main medical analysis team function with specialized medical agents"""
    run_id: Optional[str] = None
    
    # Clinical Diagnostic Agent
    clinical_diagnostic_agent = Agent(
        knowledge=medical_knowledge_base,
        name="Clinical Diagnostic Agent",
        role="Primary diagnostic reasoning specialist with expertise in differential diagnosis and clinical assessment",
        search_knowledge=True,
        tools=[medical_rag_tool_query, medical_safety_disclaimer],
        model=OpenAIChat("gpt-4.1"),
        instructions=[
            "Focus on evidence-based diagnostic reasoning",
            "Consider differential diagnoses systematically",
            "Always include clinical assessment guidelines",
            "Emphasize the importance of proper medical evaluation",
        ],
        show_tool_calls=True,
        debug_mode=debug,
    )
    
    # Medical Research Agent
    medical_research_agent = Agent(
        name="Medical Research Agent",
        role="Medical literature analysis specialist focused on evidence-based medicine and latest research findings",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[DuckDuckGoTools(), medical_rag_tool_query],
        instructions=[
            "Search for peer-reviewed medical literature",
            "Focus on evidence-based medicine principles",
            "Include publication dates and study quality",
            "Always cite reputable medical sources",
        ],
        add_datetime_to_instructions=True,
        debug_mode=debug,
    )
    
    # Pharmacology Agent
    pharmacology_agent = Agent(
        name="Pharmacology Agent",
        role="Drug interactions, dosing, and pharmaceutical safety specialist",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[DuckDuckGoTools(), medical_rag_tool_query],
        instructions=[
            "Analyze drug interactions and contraindications",
            "Provide dosing guidelines and safety information",
            "Consider pharmacokinetics and pharmacodynamics",
            "Emphasize adverse effects and monitoring requirements",
        ],
        add_datetime_to_instructions=True,
        debug_mode=debug,
    )
    
    # Specialist Consultation Agent
    specialist_consultation_agent = Agent(
        name="Specialist Consultation Agent",
        role="Multi-specialty medical expertise covering cardiology, neurology, endocrinology, and other specialties",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[DuckDuckGoTools(), medical_rag_tool_query],
        instructions=[
            "Provide specialty-specific medical insights",
            "Reference specialty guidelines and procedures",
            "Consider multidisciplinary approaches",
            "Highlight when specialist referral is needed",
        ],
        add_datetime_to_instructions=True,
        debug_mode=debug,
    )
    
    # Patient Safety Agent
    patient_safety_agent = Agent(
        name="Patient Safety Agent",
        role="Risk assessment and patient safety protocols specialist",
        model=OpenAIChat(id="gpt-4.1"),
        tools=[medical_safety_disclaimer, medical_rag_tool_query],
        instructions=[
            "Assess patient safety risks and considerations",
            "Provide safety protocols and quality measures",
            "Identify potential safety concerns",
            "Always include appropriate medical disclaimers",
        ],
        add_datetime_to_instructions=True,
        debug_mode=debug,
    )

    # Medical Reasoning Team
    medical_reasoning_team = Team(
        user_id=user,
        name="Medical Analysis Team",
        mode="coordinate",
        model=OpenAIChat(id="o4-mini-2025-04-16"),
        members=[
            clinical_diagnostic_agent,
            medical_research_agent,
            pharmacology_agent,
            specialist_consultation_agent,
            patient_safety_agent,
        ],
        tools=[ReasoningTools(add_instructions=True), medical_safety_disclaimer],
        instructions=[
            "Collaborate to provide comprehensive medical analysis and insights",
            "Always prioritize patient safety and evidence-based medicine",
            "Consider multiple medical perspectives and specialties",
            "Include appropriate medical disclaimers and safety warnings",
            "Use structured medical reasoning and differential diagnosis approaches",
            "Reference peer-reviewed medical literature and clinical guidelines",
            "Emphasize the importance of professional medical consultation",
            "Present findings in a clear, medically accurate format",
            "Always include the medical safety disclaimer in final output",
        ],
        markdown=True,
        show_members_responses=True,
        memory=medical_memory,
        enable_agentic_memory=True,
        enable_user_memories=True,
        storage=medical_storage,
        add_history_to_messages=True,
        num_history_runs=3,
        enable_agentic_context=True,
        add_datetime_to_instructions=True,
        debug_mode=debug,
        success_criteria="The medical team has provided a comprehensive, evidence-based medical analysis with appropriate safety considerations, literature references, and clear recommendations while emphasizing the need for professional medical consultation.",
    )
    
    if run_id is None:
        run_id = medical_reasoning_team.run_id
        print(f"Started Medical Analysis Run: {run_id}\n")
    else:
        print(f"Continuing Medical Analysis Run: {run_id}\n")

    if tui:
        medical_reasoning_team.print_response(message)
    else:
        response = medical_reasoning_team.run(message)
        print(response.content)
