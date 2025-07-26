# Conversation Intelligence System

A comprehensive AI-powered system that combines audio transcription, vector database storage, and multi-agent teams for financial analysis with persistent memory capabilities.

## Features

- >à **Multi-Agent Financial Analysis** with web search, RAG, and market data
- <µ **Audio/Video Transcription** with AssemblyAI integration
- = **RAG System** with Qdrant vector database and semantic search
- =¾ **Persistent Memory** with SQLite-based user memory storage
- <™ **Real-time Streaming** microphone transcription
- =Ê **Market Data Integration** via YFinance tools

## Installation

```bash
# Setup environment
make setup
uv sync --all-extras
uv run pre-commit install

# Set environment variables
export OPENAI_API_KEY=your_openai_key
export ASSEMBLY_AI_API_KEY=your_assemblyai_key
export QDRANT_URL=http://localhost:6333
```

## Command Line Usage

### Financial Analysis Commands

#### Basic Financial Analysis (TUI Mode - Default)
```bash
# Run financial analysis with full TUI interface and debug mode
uv run python cli.py finance "What's the current market outlook for NVIDIA?"

# Specify user for memory persistence
uv run python cli.py finance "Analyze Tesla stock performance" --user john_doe
```

#### Financial Analysis with Output Control
```bash
# Enable debug mode and TUI (defaults)
uv run python cli.py finance "Portfolio analysis for tech stocks" --debug --tui

# Disable debug mode but keep TUI
uv run python cli.py finance "Risk assessment for cryptocurrency" --no-debug --tui

# Text-only output without TUI formatting
uv run python cli.py finance "Market sentiment analysis" --debug --no-tui

# Minimal output - no debug, no TUI
uv run python cli.py finance "Quick stock price check for AAPL" --no-debug --no-tui
```
#### Medical Analysis with Output Control
```bash
uv run python cli.py medical "What did my docter say mean by the term 'dyspnea' in my case?" --debug --tui


```
All the Flags are applicalble to both the medical and financial analysis

#### Advanced Financial Queries
```bash
# Complex investment analysis
uv run python cli.py finance "Create a diversified portfolio with NVIDIA, Netflix, NASDAQ, Nifty 50, and P&G. Analyze risks and expected returns." --user investor_pro

# Market comparison
uv run python cli.py finance "Compare S&P 500 vs NASDAQ performance over last 6 months" --debug --tui

# Sector analysis
uv run python cli.py finance "Analyze semiconductor sector trends and top performers" --user tech_analyst
```

### Memory System Commands

#### Query Memories (TUI Mode - Default)
```bash
# Query user memories with full interface
uv run python cli.py memory query --query "What investments did I discuss last week?"

# List all stored memories
uv run python cli.py memory list

# User-specific memory queries
uv run python cli.py memory query --query "My food preferences" --user john_doe
```

#### Memory Commands with Output Control
```bash
# Query with debug and TUI (defaults)
uv run python cli.py memory query --query "Past investment decisions" --debug --tui

# Text-only memory output
uv run python cli.py memory query --query "Previous financial discussions" --no-tui

# Minimal memory listing
uv run python cli.py memory list --no-debug --no-tui

# User-specific memory with custom output
uv run python cli.py memory query --query "Investment strategy preferences" --user trader_joe --no-debug --tui
```

### Transcription Commands

#### Audio/Video Transcription
```bash
# Transcribe local audio file
uv run python cli.py transcribe "./audio/meeting.mp3"

# Transcribe YouTube video
uv run python cli.py transcribe "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Transcribe and save to specific file
uv run python cli.py transcribe "./audio/interview.wav" --output "./transcripts/interview.txt"

# Transcribe and inject into RAG system
uv run python cli.py transcribe "./audio/financial_report.mp3" --inject-rag --rag-name "Q4 Financial Report"

# YouTube transcription with RAG injection
uv run python cli.py transcribe "https://youtu.be/example" --inject-rag --rag-name "Market Analysis Video"
```

#### Real-time Streaming
```bash
# Start real-time microphone transcription
uv run python cli.py stream
```

### File Operations

#### Save and Process Files
```bash
# Save text to file
uv run python cli.py save --text "Important notes here" --output "notes.txt"

# Read and process single file
uv run python cli.py save --file "./documents/report.pdf" --output "processed_report.txt"

# Process entire folder
uv run python cli.py save --folder "./documents" --output "combined_docs.txt" --output-dir "./processed"
```

### Parameter Combinations

#### Debug Mode Examples
```bash
# Maximum verbosity - debug ON, TUI ON (shows all agent interactions, tool calls, and rich formatting)
uv run python cli.py finance "Market analysis" --debug --tui

# Debug ON, TUI OFF (detailed logs but plain text output)
uv run python cli.py finance "Stock analysis" --debug --no-tui

# Debug OFF, TUI ON (clean interface, rich formatting, minimal logs)
uv run python cli.py finance "Quick price check" --no-debug --tui

# Minimal mode - debug OFF, TUI OFF (plain text, minimal output)
uv run python cli.py finance "Simple query" --no-debug --no-tui
```

#### User-specific Workflows
```bash
# Professional trader setup
uv run python cli.py finance "Advanced options analysis" --user pro_trader --debug --tui

# Quick checks for casual investor
uv run python cli.py finance "Portfolio overview" --user casual_investor --no-debug --no-tui

# Research analyst with detailed logging
uv run python cli.py finance "Comprehensive sector analysis" --user research_analyst --debug --no-tui
```

## Python API Usage

### Direct Function Calls

```python
from agent.analysis_engine import finance_agent, memory_agent_query

# TUI mode with debug (default behavior)
finance_agent("Analyze NVIDIA stock", user="trader", debug=True, tui=True)

# Text-only output
finance_agent("Quick market update", debug=False, tui=False)

# Memory queries
memory_agent_query("What were my past investments?", debug=True, tui=True)
memory_agent_query("Investment history", debug=False, tui=False)
```

### Test Agents

```python
from agent.test import json_mode_agent_test, structured_output_agent_test

# Movie script generation with full TUI
json_mode_agent_test("Tokyo", debug=True, tui=True)

# Plain text output for structured data
structured_output_agent_test("Ancient Rome", debug=False, tui=False)
```

## Output Mode Comparison

| Mode | Debug | TUI | Use Case | Output Style |
|------|-------|-----|----------|--------------|
| **Default** |  |  | Development, Learning | Rich formatting, full debug info, agent interactions |
| **Production** | L |  | End users | Clean interface, professional output |
| **Logging** |  | L | CI/CD, Scripts | Detailed logs, plain text |
| **Minimal** | L | L | Automation, APIs | Plain text, essential output only |

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-your-openai-key
ASSEMBLY_AI_API_KEY=your-assemblyai-key

# Optional (with defaults)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-qdrant-key  # if using Qdrant Cloud
```

## Development Commands

```bash
# Testing
make test                    # Run all tests
make test-unit              # Unit tests only
make test-integration       # Integration tests only

# Code Quality
make lint                  # Linting and security checks
make format               # Code formatting
make type-check          # Type checking

# Application
uv run python main.py     # Run main application script
```

## Architecture

- **Agent Framework**: Agno library for multi-agent coordination
- **Vector Database**: Qdrant for semantic search and RAG
- **Memory**: SQLite-based persistent storage
- **Transcription**: AssemblyAI with streaming support
- **Models**: OpenAI GPT-4 for reasoning, Ollama for embeddings
- **CLI**: Typer with Rich formatting for beautiful terminal output

