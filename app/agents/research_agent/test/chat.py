#!/usr/bin/env python3
"""Simple command-line chat interface for the Research Agent."""

import asyncio
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# Load environment variables
load_dotenv()

from app.agents.research_agent import ResearchAgent

console = Console()


def display_welcome():
    """Display welcome message."""
    console.print(Panel(
        "[bold]🔍 Research Agent CLI Chat[/bold]\n\n"
        "• Ask questions and get answers from documents + web search\n" 
        "• Type 'quit' or 'exit' to leave\n"
        "• Type 'help' for more commands\n\n"
        "[dim]Powered by LangGraph + pgvector + Tavily[/dim]",
        border_style="cyan"
    ))


def display_stats(metadata):
    """Display research statistics."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("", style="dim")
    table.add_column("", style="cyan", justify="right")
    
    table.add_row("📄 Documents:", str(metadata.get('num_documents', 0)))
    table.add_row("🌐 Web Results:", str(metadata.get('num_web_results', 0)))
    
    sources = metadata.get('sources_used', [])
    if sources:
        table.add_row("📊 Sources:", ', '.join(sources))
    
    console.print(table)


async def chat_loop():
    """Main chat loop."""
    display_welcome()
    
    # Initialize agent
    agent = ResearchAgent(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_documents=5,
        max_web_results=3,
        enable_kg=False,  # Disabled for stability
        enable_web_search=True
    )
    
    console.print("[green]✓ Research Agent ready![/green]\n")
    
    async with agent:
        while True:
            try:
                # Get user input
                query = input("\n💬 Ask me anything: ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    break
                
                elif query.lower() == 'help':
                    console.print(Panel(
                        "[bold]Available Commands:[/bold]\n\n"
                        "• [cyan]help[/cyan] - Show this help\n"
                        "• [cyan]quit/exit[/cyan] - Exit chat\n\n"
                        "[bold]Example Questions:[/bold]\n"
                        "• What is LangGraph?\n"
                        "• Latest AI developments\n"
                        "• How does vector search work?",
                        title="Help",
                        border_style="blue"
                    ))
                    continue
                
                # Process research query
                console.print(f"\n[yellow]🔍 Researching: {query}[/yellow]")
                
                with console.status("[green]Searching..."):
                    result = await agent.research(query)
                
                # Display answer
                console.print("\n[bold green]📝 Answer:[/bold green]")
                console.print(Markdown(result["answer"]))
                
                # Display stats
                console.print("\n" + "─" * 50)
                display_stats(result["metadata"])
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'quit' to exit gracefully.[/yellow]")
                continue
            except EOFError:
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]❌ Error: {e}[/red]")
                continue


def main():
    """Entry point for the chat application."""
    if len(sys.argv) > 1:
        # Single query mode
        query = ' '.join(sys.argv[1:])
        
        async def single_query():
            agent = ResearchAgent(
                model="gpt-3.5-turbo",
                enable_kg=False,
                enable_web_search=True
            )
            
            console.print(f"[yellow]🔍 Researching: {query}[/yellow]")
            
            async with agent:
                result = await agent.research(query)
                
                console.print("\n[bold green]Answer:[/bold green]")
                console.print(Markdown(result["answer"]))
                
                display_stats(result["metadata"])
        
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