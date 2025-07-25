#!/usr/bin/env python3
"""Terminal launcher for C9S Agent with beautiful CLI interface."""

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up environment
from dotenv import load_dotenv
load_dotenv()

console = Console()


def display_banner():
    """Display a cool banner for C9S Agent."""
    banner_text = Text()
    banner_text.append("  ╔════════════════════════════════════════╗\n", style="cyan")
    banner_text.append("  ║                                        ║\n", style="cyan")
    banner_text.append("  ║    ", style="cyan")
    banner_text.append("🚀 C9S Agent Terminal Chat", style="bold white")
    banner_text.append("     ║\n", style="cyan")
    banner_text.append("  ║                                        ║\n", style="cyan")
    banner_text.append("  ║    ", style="cyan")
    banner_text.append("Powered by Claude + LangGraph", style="dim white")
    banner_text.append("      ║\n", style="cyan")
    banner_text.append("  ║                                        ║\n", style="cyan")
    banner_text.append("  ╚════════════════════════════════════════╝", style="cyan")
    
    console.print(Align.center(banner_text))
    console.print()


def check_environment():
    """Check if all required environment variables are set."""
    console.print("[yellow]⚙️ Checking environment configuration...[/yellow]")
    
    required_vars = {
        "ANTHROPIC_API_KEY": "Claude LLM access",
        "TAVILY_API_KEY": "Web search functionality", 
        "JIRA_URL": "JIRA integration",
        "JIRA_USERNAME": "JIRA authentication",
        "JIRA_API_KEY": "JIRA API access"
    }
    
    optional_vars = {
        "LANGSMITH_API_KEY": "LangSmith tracing",
        "DATABASE_URL": "PostgreSQL memory storage"
    }
    
    all_good = True
    
    console.print("\n[bold]Required Configuration:[/bold]")
    for var, description in required_vars.items():
        if os.getenv(var):
            console.print(f"  ✅ {var}: [green]Configured[/green] - {description}")
        else:
            console.print(f"  ❌ {var}: [red]Missing[/red] - {description}")
            all_good = False
    
    console.print("\n[bold]Optional Configuration:[/bold]")
    for var, description in optional_vars.items():
        if os.getenv(var):
            console.print(f"  ✅ {var}: [green]Configured[/green] - {description}")
        else:
            console.print(f"  ⚠️  {var}: [yellow]Not set[/yellow] - {description}")
    
    return all_good


def display_features():
    """Display C9S Agent features."""
    features = Panel(
        "[bold]🎯 Features:[/bold]\n\n"
        "• [purple]Claude 3.5 Sonnet[/purple] - Advanced reasoning and conversation\n"
        "• [green]Tavily Web Search[/green] - Real-time information retrieval\n"
        "• [yellow]JIRA MCP Integration[/yellow] - Ticket management and queries\n"
        "• [blue]Human-in-the-Loop[/blue] - Interactive decision making\n"
        "• [cyan]LangGraph Workflow[/cyan] - Structured multi-step processing\n"
        "• [magenta]LangSmith Tracing[/magenta] - Full observability and debugging\n\n"
        "[bold]🎯 Use Cases:[/bold]\n\n"
        "• Query JIRA tickets and project status\n"
        "• Get latest information from the web\n"
        "• Combine multiple data sources for insights\n"
        "• Safely execute operations with human oversight\n"
        "• Debug and trace agent decision-making\n\n"
        "[dim]Ready for intelligent conversations with enterprise integrations![/dim]",
        title="🚀 C9S Agent Capabilities",
        border_style="green"
    )
    console.print(features)


def main():
    """Main entry point for the terminal launcher."""
    try:
        display_banner()
        
        # Check environment
        if not check_environment():
            console.print("\n[red]❌ Missing required environment variables![/red]")
            console.print("[yellow]Please check your .env file and try again.[/yellow]")
            sys.exit(1)
        
        console.print("\n[green]✅ Environment check passed![/green]")
        
        # Display features
        display_features()
        
        # Ask user what they want to do
        console.print("\n[bold]Choose an option:[/bold]")
        console.print("1. 💬 [cyan]Interactive Chat Mode[/cyan] - Full featured terminal chat")
        console.print("2. ⚡ [yellow]Single Query Mode[/yellow] - Quick one-off questions")
        console.print("3. 🌐 [blue]API Server Mode[/blue] - Start the HTTP API server")
        console.print("4. 📚 [green]Help & Documentation[/green] - View agent help")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            console.print("\n[cyan]💬 Starting Interactive Chat Mode...[/cyan]")
            console.print("[dim]You can ask questions, query JIRA, search the web, and more![/dim]\n")
            
            # Run the chat interface
            chat_path = project_root / "app" / "agents" / "c9s-agent" / "test" / "chat.py"
            subprocess.run([sys.executable, str(chat_path)])
            
        elif choice == "2":
            console.print("\n[yellow]⚡ Single Query Mode[/yellow]")
            query = input("Enter your query: ").strip()
            
            if query:
                console.print(f"\n[yellow]Processing: {query}[/yellow]\n")
                chat_path = project_root / "app" / "agents" / "c9s-agent" / "test" / "chat.py"
                subprocess.run([sys.executable, str(chat_path), query])
            else:
                console.print("[red]No query provided.[/red]")
                
        elif choice == "3":
            console.print("\n[blue]🌐 Starting API Server Mode...[/blue]")
            console.print("[dim]API will be available at http://localhost:8002[/dim]\n")
            
            # Start the API server
            subprocess.run([sys.executable, "-m", "poetry", "run", "c9s-agent"])
            
        elif choice == "4":
            console.print("\n[green]📚 C9S Agent Help & Documentation[/green]")
            
            help_panel = Panel(
                "[bold]🚀 C9S Agent Documentation[/bold]\n\n"
                "[bold]Quick Start:[/bold]\n"
                "1. Ensure all environment variables are configured\n"
                "2. Choose Interactive Chat Mode for full experience\n"
                "3. Ask natural language questions\n"
                "4. Agent will route to appropriate tools (JIRA/Web)\n"
                "5. Provide feedback when prompted for human-in-the-loop\n\n"
                "[bold]Example Queries:[/bold]\n"
                "• 'Find recent bugs in the authentication system'\n"
                "• 'What are the latest AI safety research developments?'\n"
                "• 'Show me high priority tickets assigned to John'\n"
                "• 'Close all resolved tickets from last week'\n\n"
                "[bold]Human-in-the-Loop:[/bold]\n"
                "• Agent pauses for sensitive operations\n"
                "• You provide guidance or approval\n"
                "• Agent continues with your feedback\n\n"
                "[bold]Environment Variables:[/bold]\n"
                "• ANTHROPIC_API_KEY - Required for Claude LLM\n"
                "• TAVILY_API_KEY - Required for web search\n"
                "• JIRA_* - Required for JIRA integration\n"
                "• LANGSMITH_API_KEY - Optional for tracing\n\n"
                "[bold]API Endpoints (Server Mode):[/bold]\n"
                "• POST /chat - Send messages to agent\n"
                "• GET /health - Check agent status\n"
                "• GET /status - View configuration\n\n"
                "[dim]For more information, check the README.md file.[/dim]",
                title="📖 Help & Documentation",
                border_style="blue"
            )
            console.print(help_panel)
            
        else:
            console.print("[red]Invalid choice. Please select 1-4.[/red]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye! 👋[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()