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


def display_main_menu(total_cost: float = 0.0):
    """Display the main menu for C9S Agent."""
    menu_content = (
        "[bold cyan]🚀 C9S Agent Terminal[/bold cyan]\n\n"
        "[bold]1.[/bold] 💬 [cyan]Interactive Chat[/cyan]\n"
        "[bold]2.[/bold] ⚡ [yellow]Single Query[/yellow]\n"
        "[bold]3.[/bold] 🌐 [blue]API Server[/blue]\n"
        "[bold]4.[/bold] 📚 [green]Help[/green]\n"
        "[bold]5.[/bold] ⚙️  [magenta]Environment Check[/magenta]\n"
        "[bold]6.[/bold] 🚪 [red]Exit[/red]"
    )
    
    # Add cost information if greater than zero
    if total_cost > 0:
        menu_content += f"\n\n[dim]💰 Session Cost: ${total_cost:.6f}[/dim]"
    
    menu_panel = Panel(
        menu_content,
        title="Main Menu",
        border_style="cyan",
        width=40
    )
    console.print(Align.center(menu_panel))
    

def main():
    """Main entry point for the terminal launcher."""
    try:
        # Silent environment check
        env_ok = check_environment_silent()
        total_session_cost = 0.0
        
        while True:
            console.clear()
            display_main_menu(total_session_cost)
            
            if not env_ok:
                console.print("\n[yellow]⚠️  Some environment variables are missing[/yellow]")
            
            choice = console.input("\n[bold]Select option (1-6):[/bold] ").strip()
            
            if choice == "1":
                if not env_ok:
                    console.print("\n[red]❌ Please configure environment first (option 5)[/red]")
                    console.input("Press Enter to continue...")
                    continue
                console.print("\n[cyan]💬 Starting Interactive Chat Mode...[/cyan]")
                
                # Run chat and capture cost
                chat_cost = run_chat_with_cost_tracking()
                if chat_cost > 0:
                    total_session_cost += chat_cost
                    console.print(f"\n[green]💰 Chat session cost: ${chat_cost:.6f}[/green]")
                    console.input("Press Enter to continue...")
                
            elif choice == "2":
                if not env_ok:
                    console.print("\n[red]❌ Please configure environment first (option 5)[/red]")
                    console.input("Press Enter to continue...")
                    continue
                console.print("\n[yellow]⚡ Single Query Mode[/yellow]")
                query = console.input("Enter your query: ").strip()
                if query:
                    console.print(f"\n[yellow]Processing: {query}[/yellow]\n")
                    
                    # Run single query and capture cost
                    query_cost = run_single_query_with_cost_tracking(query)
                    if query_cost > 0:
                        total_session_cost += query_cost
                        console.print(f"\n[green]💰 Query cost: ${query_cost:.6f}[/green]")
                else:
                    console.print("[red]No query provided.[/red]")
                console.input("Press Enter to continue...")
                
            elif choice == "3":
                if not env_ok:
                    console.print("\n[red]❌ Please configure environment first (option 5)[/red]")
                    console.input("Press Enter to continue...")
                    continue
                console.print("\n[blue]🌐 Starting API Server Mode...[/blue]")
                console.print("[dim]API will be available at http://localhost:8002[/dim]\n")
                subprocess.run([sys.executable, "-m", "poetry", "run", "c9s-agent"])
                
            elif choice == "4":
                show_help()
                console.input("Press Enter to continue...")
                
            elif choice == "5":
                console.clear()
                display_banner()
                env_ok = check_environment()
                console.input("Press Enter to continue...")
                
            elif choice == "6":
                if total_session_cost > 0:
                    console.print(f"\n[bold cyan]💰 Total Session Cost: ${total_session_cost:.6f}[/bold cyan]")
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
                
            else:
                console.print("[red]Invalid choice. Please select 1-6.[/red]")
                console.input("Press Enter to continue...")
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye! 👋[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


def check_environment_silent():
    """Silently check if all required environment variables are set."""
    required_vars = [
        "ANTHROPIC_API_KEY",
        "TAVILY_API_KEY", 
        "JIRA_URL",
        "JIRA_USERNAME",
        "JIRA_API_KEY"
    ]
    
    return all(os.getenv(var) for var in required_vars)


def run_chat_with_cost_tracking():
    """Run interactive chat and return the total cost incurred."""
    import sys
    import asyncio
    import warnings
    import logging
    
    # Import C9S agent
    sys.path.insert(0, str(project_root / "app" / "agents" / "c9s-agent"))
    from c9s_agent import C9SAgent
    
    async def chat_session():
        """Run a chat session and return the cost."""
        agent = None
        total_cost = 0.0
        
        # Temporarily suppress logging for terminal sessions
        original_levels = {}
        loggers_to_suppress = [
            'app.memory.history.session_manager',
            'app.agents.c9s-agent.c9s_agent',
            'app.memory.database.database',
            'app.memory',
            'app.agents',
            '__main__',
            'root'
        ]
        
        # Also suppress root logger to catch any unspecific loggers
        root_logger = logging.getLogger()
        original_root_level = root_logger.level
        root_logger.setLevel(logging.CRITICAL)
        
        for logger_name in loggers_to_suppress:
            logger_obj = logging.getLogger(logger_name)
            original_levels[logger_name] = logger_obj.level
            logger_obj.setLevel(logging.CRITICAL)
        
        try:
            # Initialize agent
            config = {
                "model": os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
                "temperature": float(os.getenv("CLAUDE_TEMPERATURE", "0.1")),
                "tavily_api_key": os.getenv("TAVILY_API_KEY"),
                "jira_mcp_path": os.getenv("JIRA_MCP_PATH"),
                "enable_human_loop": os.getenv("ENABLE_HUMAN_LOOP", "true").lower() == "true",
                "postgres_connection_string": os.getenv("DATABASE_URL")
            }
            
            agent = C9SAgent(**config)
            
            # Use proper async context management
            async with agent:
                console.print("\n[bold green]🤖 C9S Agent Chat Session Started[/bold green]")
                console.print("[dim]Type 'exit', 'quit', or 'q' to return to main menu[/dim]")
                console.print("[yellow]ℹ️  Note: Memory and document search may be limited[/yellow]\n")
                
                while True:
                    try:
                        # Get user input
                        query = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
                        
                        # Handle exit commands
                        if query.lower() in ['quit', 'exit', 'q']:
                            break
                            
                        if not query:
                            continue
                            
                        # Process query with error suppression
                        console.print("\n[yellow]🔄 Processing...[/yellow]")
                        
                        # Suppress runtime warnings during processing
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=RuntimeWarning)
                            
                            result = await agent.process_query(
                                query=query,
                                user_id="terminal_user",
                                profile_id="default",
                                session_id=None
                            )
                        
                        # Display response
                        console.print(f"\n[bold blue]Agent:[/bold blue] {result['answer']}")
                        
                    except KeyboardInterrupt:
                        console.print("\n[yellow]Use 'quit' to exit gracefully.[/yellow]")
                        continue
                    except EOFError:
                        break
                    except Exception as e:
                        console.print(f"\n[red]⚠️  Processing error occurred[/red]")
                        continue
                
                # Get cost before context manager exits
                total_cost = agent.get_total_cost()
                
        except Exception as e:
            console.print(f"\n[red]⚠️  Unable to initialize chat session[/red]")
            return 0.0
        finally:
            # Restore original logging levels
            root_logger.setLevel(original_root_level)
            for logger_name, original_level in original_levels.items():
                logging.getLogger(logger_name).setLevel(original_level)
        
        return total_cost
    
    # Suppress warnings for the entire async run
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return asyncio.run(chat_session())


def run_single_query_with_cost_tracking(query: str):
    """Run a single query and return the cost incurred."""
    import sys
    import asyncio
    import warnings
    import logging
    
    # Import C9S agent
    sys.path.insert(0, str(project_root / "app" / "agents" / "c9s-agent"))
    from c9s_agent import C9SAgent
    
    async def single_query():
        """Process a single query and return the cost."""
        agent = None
        total_cost = 0.0
        
        # Temporarily suppress logging for terminal sessions
        original_levels = {}
        loggers_to_suppress = [
            'app.memory.history.session_manager',
            'app.agents.c9s-agent.c9s_agent',
            'app.memory.database.database',
            'app.memory',
            'app.agents',
            '__main__',
            'root'
        ]
        
        # Also suppress root logger to catch any unspecific loggers
        root_logger = logging.getLogger()
        original_root_level = root_logger.level
        root_logger.setLevel(logging.CRITICAL)
        
        for logger_name in loggers_to_suppress:
            logger_obj = logging.getLogger(logger_name)
            original_levels[logger_name] = logger_obj.level
            logger_obj.setLevel(logging.CRITICAL)
        
        try:
            # Initialize agent
            config = {
                "model": os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
                "temperature": float(os.getenv("CLAUDE_TEMPERATURE", "0.1")),
                "tavily_api_key": os.getenv("TAVILY_API_KEY"),
                "jira_mcp_path": os.getenv("JIRA_MCP_PATH"),
                "enable_human_loop": os.getenv("ENABLE_HUMAN_LOOP", "true").lower() == "true",
                "postgres_connection_string": os.getenv("DATABASE_URL")
            }
            
            agent = C9SAgent(**config)
            
            # Use proper async context management
            async with agent:
                # Process query with error suppression
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    
                    result = await agent.process_query(
                        query=query,
                        user_id="terminal_user",
                        profile_id="default",
                        session_id=None
                    )
                
                # Display response
                console.print(f"\n[bold blue]Agent:[/bold blue] {result['answer']}")
                
                # Get cost before context manager exits
                total_cost = agent.get_total_cost()
                
        except Exception as e:
            console.print(f"\n[red]⚠️  Unable to process query[/red]")
            return 0.0
        finally:
            # Restore original logging levels
            root_logger.setLevel(original_root_level)
            for logger_name, original_level in original_levels.items():
                logging.getLogger(logger_name).setLevel(original_level)
        
        return total_cost
    
    # Suppress warnings for the entire async run
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return asyncio.run(single_query())


def show_help():
    """Display help information."""
    console.clear()
    
    # Main documentation panel
    help_panel = Panel(
        "[bold cyan]🤖 What is C9S Agent?[/bold cyan]\n"
        "An intelligent AI assistant that combines Claude 3.5 Sonnet with enterprise tools\n"
        "to help with project management, research, and workflow automation.\n\n"
        
        "[bold yellow]🛠️  Available Tools:[/bold yellow]\n"
        "• [blue]Claude 3.5 Sonnet[/blue] - Advanced reasoning and conversation\n"
        "• [green]Tavily Web Search[/green] - Real-time internet information\n"
        "• [red]JIRA Integration[/red] - Ticket management via MCP protocol\n"
        "• [purple]LangGraph Workflow[/purple] - Multi-step task orchestration\n"
        "• [magenta]Human-in-the-Loop[/magenta] - Interactive decision making\n\n"
        
        "[bold green]🧠 Memory System:[/bold green]\n"
        "• [cyan]Vector Database[/cyan] - Semantic search through documents\n"
        "• [yellow]Chat History[/yellow] - Session-based conversation memory\n"
        "• [blue]Knowledge Graph[/blue] - Structured relationship mapping\n"
        "• [white]PostgreSQL + pgvector[/white] - Persistent storage backend\n\n"
        
        "[bold red]🎯 Use Cases:[/bold red]\n"
        "• Query and manage JIRA tickets intelligently\n"
        "• Research latest information from the web\n"
        "• Combine multiple data sources for insights\n"
        "• Automate workflows with safety checks\n"
        "• Maintain context across conversations\n\n"
        
        "[dim]💡 The agent automatically chooses the right tools for your task[/dim]",
        title="📖 C9S Agent Documentation",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(help_panel)
    
    # Quick start panel
    console.print()
    quick_start_panel = Panel(
        "[bold]1.[/bold] Configure environment (option 5 in main menu)\n"
        "[bold]2.[/bold] Choose Interactive Chat for full experience\n"
        "[bold]3.[/bold] Ask natural language questions - agent handles the rest!\n\n"
        
        "[bold cyan]Example Queries:[/bold cyan]\n"
        "[dim]Project Management:[/dim]\n"
        "• 'Show me all critical bugs assigned to the auth team'\n"
        "• 'Create a summary of completed tickets this sprint'\n\n"
        "[dim]Research & Information:[/dim]\n"
        "• 'What are the latest developments in AI safety?'\n"
        "• 'Find best practices for microservices architecture'\n\n"
        "[dim]Workflow Automation:[/dim]\n"
        "• 'Close all resolved tickets from last week'\n"
        "• 'Update ticket priorities based on security findings'",
        title="🚀 Quick Start Guide",
        border_style="green",
        padding=(1, 2)
    )
    console.print(quick_start_panel)
    
    console.print("\n[dim]Press Enter to return to main menu...[/dim]")


if __name__ == "__main__":
    main()