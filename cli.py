#!/usr/bin/env python3
"""
Unified CLI tool for Conversation Intelligence System
Provides command-line access to all agents and functionality.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

app = typer.Typer(
    name="intelligence-cli",
    help="🧠 Unified Conversation Intelligence CLI Tool",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


def check_environment():
    """Check if required environment variables are set."""
    required_vars = ["OPENAI_API_KEY", "ASSEMBLY_AI_API_KEY", "QDRANT_URL"]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        console.print(
            f"[red]❌ Missing environment variables: {', '.join(missing_vars)}[/red]"
        )
        console.print("[yellow]💡 Please set these in your .env file[/yellow]")
        return False
    return True


@app.command()
def finance(
    query: str = typer.Argument(..., help="Financial analysis query"),
    user: str = typer.Option("user", help="User identifier for memory storage"),
    output: Optional[str] = typer.Option(None, help="Output file path (optional)"),
    debug: bool = typer.Option(True, "--debug/--no-debug", help="Enable/disable debug mode"),
    tui: bool = typer.Option(True, "--tui/--no-tui", help="Enable/disable full TUI mode (vs text-only)"),
):
    """
    🏦 Run financial analysis with multi-agent team.

    Combines web search, RAG knowledge, and market data for comprehensive analysis.
    """
    if not check_environment():
        raise typer.Exit(1)

    try:
        from agent.analysis_engine import finance_agent

        console.print(f"[green]🔍 Running financial analysis for: {query}[/green]")
        console.print(f"[blue]👤 User: {user}[/blue]")

        with console.status("[bold green]Analyzing..."):
            finance_agent(query, user, debug, tui)

        console.print("[green]✅ Financial analysis completed[/green]")

        if output:
            console.print(f"[blue]💾 Results saved to: {output}[/blue]")

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error during analysis: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def medical(
    query: str = typer.Argument(..., help="Medical analysis query"),
    user: str = typer.Option("medical_user", help="User identifier for memory storage"),
    output: Optional[str] = typer.Option(None, help="Output file path (optional)"),
    debug: bool = typer.Option(True, "--debug/--no-debug", help="Enable/disable debug mode"),
    tui: bool = typer.Option(True, "--tui/--no-tui", help="Enable/disable full TUI mode (vs text-only)"),
):
    """
    🏥 Run medical analysis with multi-agent team.

    Provides clinical diagnostic reasoning, research, pharmacology, and safety analysis.
    """
    if not check_environment():
        raise typer.Exit(1)

    try:
        from agent.medical_analysis_engine import medical_analysis_team

        console.print(f"[green]🔍 Running medical analysis for: {query}[/green]")
        console.print(f"[blue]👤 User: {user}[/blue]")
        console.print("[yellow]⚠️  Medical information is for educational purposes only[/yellow]")

        with console.status("[bold green]Analyzing..."):
            medical_analysis_team(query, user, debug, tui)

        console.print("[green]✅ Medical analysis completed[/green]")

        if output:
            console.print(f"[blue]💾 Results saved to: {output}[/blue]")

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error during analysis: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def legal(
    query: str = typer.Argument(..., help="Legal analysis query"),
    user: str = typer.Option("legal_user", help="User identifier for memory storage"),
    output: Optional[str] = typer.Option(None, help="Output file path (optional)"),
    debug: bool = typer.Option(True, "--debug/--no-debug", help="Enable/disable debug mode"),
    tui: bool = typer.Option(True, "--tui/--no-tui", help="Enable/disable full TUI mode (vs text-only)"),
):
    """
    ⚖️ Run legal analysis with multi-agent team.

    Provides legal research, contract analysis, regulatory compliance, and risk assessment.
    """
    if not check_environment():
        raise typer.Exit(1)

    try:
        from agent.legal_analysis_engine import legal_analysis_team

        console.print(f"[green]🔍 Running legal analysis for: {query}[/green]")
        console.print(f"[blue]👤 User: {user}[/blue]")
        console.print("[yellow]⚠️  Legal information is for educational purposes only - not legal advice[/yellow]")

        with console.status("[bold green]Analyzing..."):
            legal_analysis_team(query, user, debug, tui)

        console.print("[green]✅ Legal analysis completed[/green]")

        if output:
            console.print(f"[blue]💾 Results saved to: {output}[/blue]")

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error during analysis: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def memory(
    action: str = typer.Argument(..., help="Action: 'query' or 'list'"),
    query: Optional[str] = typer.Option(
        None, help="Memory query (required for 'query' action)"
    ),
    user: str = typer.Option("user", help="User identifier"),
    medical: bool = typer.Option(False, help="Use medical memory database"),
    legal: bool = typer.Option(False, help="Use legal memory database"),
    debug: bool = typer.Option(True, "--debug/--no-debug", help="Enable/disable debug mode"),
    tui: bool = typer.Option(True, "--tui/--no-tui", help="Enable/disable full TUI mode (vs text-only)"),
):
    """
    🧠 Query or list stored memories.

    Access your stored memories and conversation history.
    """
    if not check_environment():
        raise typer.Exit(1)

    try:
        if medical:
            from agent.medical_analysis_engine import medical_memory_agent_query
            memory_function = medical_memory_agent_query
            console.print("[blue]🏥 Using medical memory database[/blue]")
        elif legal:
            from agent.legal_analysis_engine import legal_memory_agent_query
            memory_function = legal_memory_agent_query
            console.print("[blue]⚖️ Using legal memory database[/blue]")
        else:
            from agent.analysis_engine import memory_agent_query
            memory_function = memory_agent_query

        if action == "query":
            if not query:
                console.print("[red]❌ Query is required for 'query' action[/red]")
                raise typer.Exit(1)

            console.print(f"[green]🔍 Querying memories: {query}[/green]")
            with console.status("[bold green]Searching memories..."):
                memory_function(query, debug, tui)

        elif action == "list":
            console.print("[green]📋 Listing all memories[/green]")
            memory_function("Tell me all about the past memory", debug, tui)
        else:
            console.print(
                f"[red]❌ Unknown action: {action}. Use 'query' or 'list'[/red]"
            )
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error accessing memories: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def transcribe(
    source: str = typer.Argument(..., help="Source: file path or YouTube URL"),
    output: Optional[str] = typer.Option(None, help="Output file path (optional)"),
    inject_rag: bool = typer.Option(False, help="Inject transcript into RAG system"),
    rag_name: Optional[str] = typer.Option(None, help="Name for RAG injection"),
):
    """
    🎵 Transcribe audio files or YouTube videos.

    Supports various audio formats and YouTube URLs with optional RAG injection.
    """
    if not check_environment():
        raise typer.Exit(1)

    try:
        from assemblyAi.tts import get_transcript_from_file, get_transcript_from_youtube
        from file_saver.save import save_text_to_file
        from qdrant_rag.rag_client import RagClient

        transcript = None
        source_name = None

        if source.startswith(("http://", "https://", "youtube.com", "youtu.be")):
            console.print(f"[green]📺 Transcribing YouTube video: {source}[/green]")
            with console.status("[bold green]Downloading and transcribing..."):
                transcript = get_transcript_from_youtube(source)
                source_name = rag_name or "YouTube Video Transcript"
        else:
            if not Path(source).exists():
                console.print(f"[red]❌ File not found: {source}[/red]")
                raise typer.Exit(1)

            console.print(f"[green]🎵 Transcribing audio file: {source}[/green]")
            with console.status("[bold green]Transcribing..."):
                transcript = get_transcript_from_file(source)
                source_name = rag_name or Path(source).stem

        console.print("[green]✅ Transcription completed[/green]")
        console.print(
            f"[blue]📝 Transcript length: {len(transcript)} characters[/blue]"
        )

        if output:
            save_text_to_file(transcript, output)
            console.print(f"[blue]💾 Transcript saved to: {output}[/blue]")

        if inject_rag:
            console.print("[green]🔗 Injecting transcript into RAG system[/green]")
            rag_client = RagClient()
            rag_client.inject_text(transcript, name=source_name)
            console.print("[green]✅ Transcript injected into RAG[/green]")

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error during transcription: {e}[/red]")
        raise typer.Exit(1)



@app.command()
def converse(
    team: str = typer.Option("finance", help="Agent team to use: 'finance', 'medical', or 'legal'"),
    user: str = typer.Option("user", help="User identifier for memory storage"),
    debug: bool = typer.Option(True, "--debug/--no-debug", help="Enable/disable debug mode"),
    tui: bool = typer.Option(True, "--tui/--no-tui", help="Enable/disable full TUI mode (vs text-only)"),
):
    """
    🗣️ Start a conversational session with AI agent teams using voice input.

    Real-time voice-to-text transcription with intelligent agent responses.
    """
    if not check_environment():
        raise typer.Exit(1)

    try:
        from agent.conversation_handler import create_conversation_handler, ConversationManager

        # Validate team type
        if not ConversationManager.validate_team_type(team):
            console.print(f"[red]❌ Invalid team type: {team}[/red]")
            ConversationManager.display_team_info(console)
            raise typer.Exit(1)

        # Display team information
        ConversationManager.display_team_info(console)
        
        # Create and start conversation handler
        conversation_handler = create_conversation_handler(team, user, debug, tui)
        conversation_handler.start_conversation()

    except KeyboardInterrupt:
        console.print("\n[yellow]🛑 Conversation stopped by user[/yellow]")
    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error during conversation: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def stream():
    """
    🎙️ Real-time audio transcription from microphone.

    Start streaming transcription session.
    """
    if not check_environment():
        raise typer.Exit(1)

    try:
        from assemblyAi.streamer import main as stream_main

        console.print("[green]🎙️  Starting real-time transcription[/green]")
        console.print("[blue]ℹ️  Press Ctrl+C to stop[/blue]")

        stream_main()

    except KeyboardInterrupt:
        console.print("\n[yellow]🛑 Streaming stopped by user[/yellow]")
    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error during streaming: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def save(
    text: Optional[str] = typer.Option(None, help="Text to save"),
    file: Optional[str] = typer.Option(None, help="File to read and save"),
    folder: Optional[str] = typer.Option(None, help="Folder to read all files from"),
    output: str = typer.Option("output.txt", help="Output file name"),
    output_dir: str = typer.Option("output", help="Output directory"),
):
    """
    💾 File operations: save text or read files/folders.

    Save text to files or read content from files/folders.
    """
    try:
        from file_saver.save import read_all_files_in_folder, save_text_to_file

        if text:
            console.print(f"[green]💾 Saving text to: {output}[/green]")
            save_text_to_file(text, output, output_dir)
            console.print("[green]✅ Text saved successfully[/green]")

        elif file:
            if not Path(file).exists():
                console.print(f"[red]❌ File not found: {file}[/red]")
                raise typer.Exit(1)

            console.print(f"[green]📄 Reading file: {file}[/green]")
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()

            save_text_to_file(content, output, output_dir)
            console.print(f"[green]✅ File content saved to: {output}[/green]")

        elif folder:
            if not Path(folder).exists():
                console.print(f"[red]❌ Folder not found: {folder}[/red]")
                raise typer.Exit(1)

            console.print(f"[green]📁 Reading all files from: {folder}[/green]")
            content = read_all_files_in_folder(folder)

            if content:
                save_text_to_file(content, output, output_dir)
                console.print(f"[green]✅ Folder content saved to: {output}[/green]")
            else:
                console.print("[yellow]⚠️  No content found in folder[/yellow]")
        else:
            console.print("[red]❌ Please provide text, file, or folder option[/red]")
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error with file operation: {e}[/red]")
        raise typer.Exit(1)


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", help="Show version and exit"
    )
):
    """
    🧠 Unified Conversation Intelligence CLI Tool

    Access all agents and functionality through a single command-line interface.
    """
    if version:
        console.print("[green]🧠 Conversation Intelligence CLI v1.0.0[/green]")
        raise typer.Exit()


if __name__ == "__main__":
    app()

