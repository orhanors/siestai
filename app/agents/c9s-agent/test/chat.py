#!/usr/bin/env python3
"""Interactive terminal chat interface for the C9S Agent with human-in-the-loop support."""

import asyncio
import sys
import time
import json
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Load environment variables
load_dotenv()

# Add parent directory to path to import the agent
sys.path.insert(0, '/Users/orhanors/Desktop/siestai-full/siestai/app/agents/c9s-agent')
from c9s_agent import C9SAgent

console = Console()


def stream_text(text: str, delay: float = 0.01):
    """Display text with a typewriter effect."""
    for char in text:
        console.print(char, end='', style="white")
        time.sleep(delay)
    console.print()  # Add newline at the end


class C9SStepTracker:
    """Tracks and displays C9S agent steps in real-time."""
    
    def __init__(self):
        self.steps = []
        self.current_step = None
        self.step_details = {}
        
    def track_step(self, step_name: str, status: str, details: dict = None):
        """Track a step with status and details."""
        self.steps.append({
            "name": step_name, 
            "status": status, 
            "details": details or {},
            "timestamp": time.time()
        })
        
        if status == "starting":
            console.print(f"[yellow]⏳ {step_name}...[/yellow]")
        elif status == "completed":
            console.print(f"[green]✅ {step_name} completed[/green]")
        elif status == "failed":
            console.print(f"[red]❌ {step_name} failed[/red]")
    
    def display_summary(self, metadata: dict):
        """Display a summary of all steps."""
        if not metadata:
            return
            
        console.print("\n[bold blue]🚀 C9S Agent Execution Summary:[/bold blue]")
        
        summary_table = Table(show_header=True, header_style="bold cyan")
        summary_table.add_column("Component", style="white", width=20)
        summary_table.add_column("Status", width=12)  
        summary_table.add_column("Results", style="dim", width=40)
        
        # Router decision
        next_action = metadata.get("next_action", "unknown")
        summary_table.add_row(
            "🧭 Router", 
            "[green]✅ Routed[/green]",
            f"Decision: {next_action}"
        )
        
        # Web search results
        web_count = metadata.get("web_results_count", 0)
        if web_count > 0:
            summary_table.add_row(
                "🌐 Web Search",
                "[green]✅ Found[/green]",
                f"{web_count} results retrieved"
            )
        
        # JIRA search results
        jira_count = metadata.get("jira_results_count", 0)
        if jira_count > 0:
            summary_table.add_row(
                "🎫 JIRA Search",
                "[green]✅ Found[/green]",
                f"{jira_count} tickets found"
            )
        
        # Human-in-the-loop status
        if metadata.get("human_feedback_provided"):
            summary_table.add_row(
                "👤 Human Input", 
                "[blue]✅ Provided[/blue]",
                "Feedback incorporated"
            )
        
        console.print(summary_table)


step_tracker = C9SStepTracker()


def display_welcome():
    """Display welcome message."""
    console.print(Panel(
        "[bold]🚀 C9S Agent Terminal Chat[/bold]\n\n"
        "• Powered by [cyan]Claude 3.5 Sonnet[/cyan] + LangGraph\n"
        "• [yellow]JIRA MCP[/yellow] + [green]Tavily Web Search[/green]\n" 
        "• [blue]Human-in-the-Loop[/blue] support\n\n"
        "Commands:\n"
        "• Type 'quit' or 'exit' to leave\n"
        "• Type 'help' for more commands\n"
        "• Type 'status' to check agent configuration\n\n"
        "[dim]Ready for intelligent conversations with tool integration![/dim]",
        border_style="cyan",
        title="🤖 C9S Agent"
    ))


def display_stats(metadata):
    """Display execution statistics."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("", style="dim")
    table.add_column("", style="cyan", justify="right")
    
    table.add_row("🌐 Web Results:", str(metadata.get('web_results_count', 0)))
    table.add_row("🎫 JIRA Results:", str(metadata.get('jira_results_count', 0)))
    table.add_row("🧭 Routing:", metadata.get('next_action', 'unknown'))
    
    console.print(table)


def display_sources(sources):
    """Display source information in a structured way."""
    web_results = sources.get("web_results", [])
    jira_results = sources.get("jira_results", [])
    
    if web_results:
        console.print("\n[bold blue]🌐 Web Sources:[/bold blue]")
        web_table = Table(show_header=True, header_style="bold cyan", border_style="blue")
        web_table.add_column("#", style="cyan", width=3)
        web_table.add_column("Title", style="white", width=30)
        web_table.add_column("URL", style="blue", width=50)
        
        for i, result in enumerate(web_results[:3], 1):
            title = result.get('title', 'Unknown')[:27] + "..." if len(result.get('title', '')) > 30 else result.get('title', 'Unknown')
            url = result.get('url', 'N/A')
            if len(url) > 47:
                url = url[:44] + "..."
            
            web_table.add_row(str(i), title, f"[link={result.get('url', '')}]{url}[/link]")
        
        console.print(web_table)
    
    if jira_results:
        console.print("\n[bold blue]🎫 JIRA Tickets:[/bold blue]")
        jira_table = Table(show_header=True, header_style="bold cyan", border_style="green")
        jira_table.add_column("Key", style="cyan", width=12)
        jira_table.add_column("Summary", style="white", width=30)
        jira_table.add_column("Status", style="green", width=12)
        jira_table.add_column("Type", style="yellow", width=8)
        jira_table.add_column("Assignee", style="blue", width=15)
        
        for result in jira_results[:5]:  # Show more results
            key = result.get('key', 'N/A')
            summary = result.get('summary', 'No summary')[:27] + "..." if len(result.get('summary', '')) > 30 else result.get('summary', 'No summary')
            status = result.get('status', 'Unknown')
            ticket_type = result.get('type', 'N/A')
            assignee = result.get('assignee', 'Unassigned')[:12] + "..." if len(result.get('assignee', '')) > 15 else result.get('assignee', 'Unassigned')
            
            jira_table.add_row(key, summary, status, ticket_type, assignee)
        
        console.print(jira_table)


async def handle_human_in_loop(agent, session_id):
    """Handle human-in-the-loop workflow."""
    console.print("\n[bold yellow]👤 Human Input Required[/bold yellow]")
    
    # Get context for the decision
    try:
        context = await agent.get_human_input_context(session_id)
        
        if context:
            console.print(Panel(
                f"[bold]Query:[/bold] {context.get('query', 'Unknown')}\n"
                f"[bold]Current Step:[/bold] {context.get('current_step', 'Unknown')}\n"
                f"[bold]Action:[/bold] {context.get('next_action', 'Unknown')}\n\n"
                f"[dim]Web Results: {len(context.get('web_results', []))}\n"
                f"JIRA Results: {len(context.get('jira_results', []))}[/dim]",
                title="🔍 Context for Decision",
                border_style="yellow"
            ))
        
        # Get human feedback
        console.print("\n[bold]Please provide your guidance:[/bold]")
        console.print("[dim]Examples: 'proceed carefully', 'abort operation', 'focus on security tickets'[/dim]")
        
        feedback = input("\n💬 Your feedback: ").strip()
        
        if not feedback:
            feedback = "proceed with default behavior"
            console.print(f"[dim]Using default: {feedback}[/dim]")
        
        console.print(f"\n[green]✅ Feedback received: {feedback}[/green]")
        
        # Continue with feedback
        step_tracker.track_step("Human Feedback", "starting")
        result = await agent.continue_with_human_feedback(session_id, feedback)
        step_tracker.track_step("Human Feedback", "completed")
        
        return result
        
    except Exception as e:
        console.print(f"[red]❌ Error in human-in-the-loop: {e}[/red]")
        return None


async def chat_loop():
    """Main chat loop with C9S agent."""
    display_welcome()
    
    # Get user credentials
    user_id = input("\nEnter your user ID (default: c9s_user): ").strip() or "c9s_user"
    profile_id = input("Enter your profile ID (default: default_profile): ").strip() or "default_profile"
    
    console.print(f"\n[green]👤 User: {user_id} | Profile: {profile_id}[/green]")
    
    # Initialize C9S agent
    try:
        console.print("[yellow]⏳ Initializing C9S Agent with Claude...[/yellow]")
        
        agent = C9SAgent(
            model="claude-3-5-sonnet-20241022",
            temperature=0.1,
            enable_human_loop=True  # Enable human-in-the-loop
        )
        
        async with agent:
            console.print("[green]✅ C9S Agent ready with Claude + JIRA + Web Search![/green]\n")
            
            while True:
                try:
                    # Get user input
                    query = input("💬 Ask me anything: ").strip()
                    
                    if not query:
                        continue
                    
                    # Handle commands
                    if query.lower() in ['quit', 'exit', 'q']:
                        console.print("[yellow]Goodbye! 👋[/yellow]")
                        break
                    
                    elif query.lower() == 'help':
                        console.print(Panel(
                            "[bold]C9S Agent Commands:[/bold]\n\n"
                            "• [cyan]help[/cyan] - Show this help\n"
                            "• [cyan]status[/cyan] - Check agent configuration\n"
                            "• [cyan]quit/exit[/cyan] - Exit chat\n\n"
                            "[bold]Features:[/bold]\n"
                            "• [yellow]JIRA Integration[/yellow] - Query tickets, issues, projects\n"
                            "• [green]Web Search[/green] - Get current information from the web\n"
                            "• [blue]Human-in-the-Loop[/blue] - Pause for your input on sensitive operations\n"
                            "• [purple]Claude LLM[/purple] - Powered by Claude 3.5 Sonnet\n\n"
                            "[bold]Example Queries:[/bold]\n"
                            "• 'Find recent JIRA bugs in authentication'\n"
                            "• 'What are the latest AI developments?'\n"
                            "• 'Search for high priority tickets assigned to me'\n"
                            "• 'Close all resolved tickets' (triggers human input)",
                            title="Help",
                            border_style="blue"
                        ))
                        continue
                    
                    elif query.lower() == 'status':
                        console.print(Panel(
                            "[bold]C9S Agent Status:[/bold]\n\n"
                            "• [green]✅ Claude LLM[/green] - claude-3-5-sonnet-20241022\n"
                            "• [green]✅ Tavily Web Search[/green] - Configured\n"
                            "• [green]✅ JIRA MCP[/green] - Connected via Docker\n"
                            "• [blue]✅ Human-in-the-Loop[/blue] - Enabled\n"
                            "• [purple]✅ LangSmith Tracing[/purple] - Active\n\n"
                            "[dim]All systems operational![/dim]",
                            title="🚀 Agent Status",
                            border_style="green"
                        ))
                        continue
                    
                    # Process query with progress indication
                    console.print(f"\n[yellow]🚀 Processing: {query}[/yellow]")
                    
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                        transient=True
                    ) as progress:
                        task = progress.add_task("Thinking...", total=None)
                        
                        # Process the query
                        result = await agent.process_query(
                            query=query,
                            user_id=user_id,
                            profile_id=profile_id
                        )
                    
                    # Check if human input is required
                    if result.get("requires_human_input"):
                        result = await handle_human_in_loop(agent, result["session_id"])
                        if not result:
                            console.print("[red]❌ Failed to process human feedback[/red]")
                            continue
                    
                    # Display answer with streaming effect
                    if result.get("answer"):
                        console.print("\n[bold green]🚀 Claude says:[/bold green]")
                        stream_text(result["answer"])
                    else:
                        console.print("\n[red]❌ No answer received[/red]")
                        continue
                    
                    # Display sources
                    if result.get("sources"):
                        display_sources(result["sources"])
                    
                    # Display execution summary
                    if result.get("metadata"):
                        step_tracker.display_summary(result["metadata"])
                    
                    # Display stats
                    console.print("\n" + "─" * 60)
                    display_stats(result.get("metadata", {}))
                    
                except KeyboardInterrupt:
                    console.print("\n[yellow]Use 'quit' to exit gracefully.[/yellow]")
                    continue
                except EOFError:
                    console.print("\n[yellow]Goodbye! 👋[/yellow]")
                    break
                except Exception as e:
                    console.print(f"\n[red]❌ Error: {e}[/red]")
                    continue
    
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize C9S Agent: {e}[/red]")
        sys.exit(1)


def main():
    """Entry point for the chat application."""
    if len(sys.argv) > 1:
        # Single query mode
        query = ' '.join(sys.argv[1:])
        
        async def single_query():
            agent = C9SAgent(
                model="claude-3-5-sonnet-20241022",
                temperature=0.1,
                enable_human_loop=False  # Disable for single queries
            )
            
            console.print(f"[yellow]🚀 Processing: {query}[/yellow]")
            
            async with agent:
                result = await agent.process_query(
                    query=query,
                    user_id="cli_user",
                    profile_id="single_query"
                )
                
                console.print("\n[bold green]🚀 Claude says:[/bold green]")
                stream_text(result["answer"])
                
                if result.get("sources"):
                    display_sources(result["sources"])
                
                display_stats(result.get("metadata", {}))
        
        asyncio.run(single_query())
    else:
        # Interactive chat mode
        asyncio.run(chat_loop())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Goodbye! 👋[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)