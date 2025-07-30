"""
Conversation Handler Module

Integrates AssemblyAI streaming transcription with agent teams for real-time conversations.
Supports both financial and medical analysis teams with memory persistence.
"""

import threading
import time
from typing import Optional, Callable, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from agno.team.team import Team

from assemblyAi.streamer import ConversationalStreamer
from agent.analysis_engine import finance_agent, create_finance_team
from agent.medical_analysis_engine import medical_analysis_team, create_medical_team
from agent.legal_analysis_engine import legal_analysis_team, create_legal_team


class ConversationHandler:
    """Handles real-time conversations between user voice input and AI agent teams."""
    
    def __init__(
        self,
        team_type: str = "finance",
        user: str = "user",
        debug: bool = True,
        tui: bool = True
    ):
        self.team_type = team_type
        self.user = user
        self.debug = debug
        self.tui = tui
        self.console = Console()
        self.conversation_active = False
        self.streamer = None
        self.agent_team = self._create_persistent_agent_team()
        
    def _create_persistent_agent_team(self) -> Team:
        """Create and return a persistent agent team instance."""
        if self.team_type == "finance":
            return create_finance_team(user=self.user, debug=self.debug)
        elif self.team_type == "medical":
            return create_medical_team(user=self.user, debug=self.debug)
        elif self.team_type == "legal":
            return create_legal_team(user=self.user, debug=self.debug)
        else:
            raise ValueError(f"Unknown team type: {self.team_type}")
    
    def _process_transcript(self, transcript: str) -> None:
        """Process transcript and send to appropriate agent team."""
        if not transcript.strip():
            return
            
        self.console.print(
            Panel(
                f"[bold blue]You said:[/bold blue] {transcript}",
                title="🎤 Voice Input",
                border_style="blue"
            )
        )
        
        # Process with agent team in a separate thread to avoid blocking
        agent_thread = threading.Thread(
            target=self._run_agent_analysis,
            args=(transcript,),
            daemon=True
        )
        agent_thread.start()
        
    def _run_agent_analysis(self, message: str) -> None:
        """Run agent analysis in separate thread."""
        try:
            self.console.print(
                Panel(
                    f"[bold yellow]Processing with {self.team_type} team...[/bold yellow]",
                    title="🤖 AI Analysis",
                    border_style="yellow"
                )
            )
            
            # Use the persistent agent team
            if self.tui:
                self.agent_team.print_response(message)
            else:
                response = self.agent_team.run(message)
                print(response.content)
            
            self.console.print(
                Panel(
                    "[bold green]Analysis complete! Ready for next input.[/bold green]",
                    title="✅ Ready",
                    border_style="green"
                )
            )
            
        except Exception as e:
            self.console.print(
                Panel(
                    f"[bold red]Error during analysis: {e}[/bold red]",
                    title="❌ Error",
                    border_style="red"
                )
            )
    
    def start_conversation(self) -> None:
        """Start the conversational session."""
        self.console.print(
            Panel(
                f"[bold green]Starting conversation with {self.team_type} team[/bold green]\n"
                f"[blue]User: {self.user}[/blue]\n"
                f"[yellow]Speak into your microphone. Press Ctrl+C to stop.[/yellow]",
                title="🗣️ Conversation Mode",
                border_style="green"
            )
        )
        
        if self.team_type == "medical":
            self.console.print(
                Panel(
                    "[bold yellow]⚠️ Medical information is for educational purposes only.\n"
                    "Always consult with healthcare professionals for medical advice.[/bold yellow]",
                    title="Medical Disclaimer",
                    border_style="yellow"
                )
            )
        elif self.team_type == "legal":
            self.console.print(
                Panel(
                    "[bold yellow]⚖️ Legal information is for educational purposes only - not legal advice.\n"
                    "No attorney-client relationship is created. Always consult qualified legal counsel.[/bold yellow]",
                    title="Legal Disclaimer",
                    border_style="yellow"
                )
            )
        
        try:
            self.conversation_active = True
            self.streamer = ConversationalStreamer(
                on_transcript_callback=self._process_transcript
            )
            
            # Start streaming in the main thread (blocks until stopped)
            self.streamer.start_streaming()
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]🛑 Conversation stopped by user[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]❌ Error during conversation: {e}[/red]")
        finally:
            self.stop_conversation()
    
    def stop_conversation(self) -> None:
        """Stop the conversational session."""
        self.conversation_active = False
        if self.streamer:
            self.streamer.stop_streaming()
        
        self.console.print(
            Panel(
                "[bold blue]Conversation session ended.[/bold blue]\n"
                "[green]Thank you for using the conversational AI system![/green]",
                title="👋 Session Complete",
                border_style="blue"
            )
        )


class ConversationManager:
    """Manages multiple conversation sessions and provides utilities."""
    
    @staticmethod
    def get_available_teams() -> Dict[str, str]:
        """Get list of available agent teams."""
        return {
            "finance": "🏦 Financial Analysis Team - Market data, investment analysis, financial planning",
            "medical": "🏥 Medical Analysis Team - Clinical diagnostics, research, pharmacology, safety",
            "legal": "⚖️ Legal Analysis Team - Legal research, contract analysis, regulatory compliance, risk assessment"
        }
    
    @staticmethod
    def validate_team_type(team_type: str) -> bool:
        """Validate if the team type is supported."""
        return team_type in ConversationManager.get_available_teams()
    
    @staticmethod
    def display_team_info(console: Console) -> None:
        """Display information about available teams."""
        teams = ConversationManager.get_available_teams()
        
        console.print(
            Panel(
                "\n".join([f"[bold]{team}[/bold]: {desc}" for team, desc in teams.items()]),
                title="🤖 Available Agent Teams",
                border_style="cyan"
            )
        )


def create_conversation_handler(
    team_type: str,
    user: str = "user",
    debug: bool = True,
    tui: bool = True
) -> ConversationHandler:
    """Factory function to create a conversation handler."""
    if not ConversationManager.validate_team_type(team_type):
        raise ValueError(
            f"Invalid team type: {team_type}. "
            f"Available teams: {list(ConversationManager.get_available_teams().keys())}"
        )
    
    return ConversationHandler(team_type, user, debug, tui)