#!/usr/bin/env python3
"""Simple command-line chat interface for the Research Agent."""

import asyncio
import sys
import time
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
# Removed unused Rich imports - using simpler display approach

# Load environment variables
load_dotenv()

from app.agents.research_agent import ResearchAgent

console = Console()


def stream_text(text: str, delay: float = 0.01):
    """Display text with a typewriter effect."""
    for char in text:
        console.print(char, end='', style="white")
        time.sleep(delay)
    console.print()  # Add newline at the end


class StepTracker:
    """Tracks and displays research agent steps in real-time."""
    
    def __init__(self):
        self.steps = []
        self.current_step = None
        self.step_details = {}
        
    async def step_callback(self, message: str, state):
        """Callback function for step tracking."""
        if "🔄 Starting:" in message:
            step_name = message.split("🔄 Starting: ")[1]
            self.current_step = step_name
            self.steps.append({"name": step_name, "status": "running", "message": message})
            console.print(f"[yellow]{message}[/yellow]")
            
        elif "✅ Completed:" in message:
            step_name = message.split("✅ Completed: ")[1].split(" (")[0]
            # Update the last step status
            for step in reversed(self.steps):
                if step["name"] == step_name and step["status"] == "running":
                    step["status"] = "completed"
                    step["message"] = message
                    break
            
            # Store step results
            if hasattr(state, 'get') and state.get("step_results"):
                self.step_details[step_name] = state["step_results"].get(step_name, {})
                
            console.print(f"[green]{message}[/green]")
            
        elif "❌ Failed:" in message:
            step_name = message.split("❌ Failed: ")[1].split(" - ")[0]
            for step in reversed(self.steps):
                if step["name"] == step_name and step["status"] == "running":
                    step["status"] = "failed"
                    step["message"] = message
                    break
            console.print(f"[red]{message}[/red]")
    
    def display_summary(self, step_timings: dict, step_results: dict):
        """Display a summary of all steps."""
        if not self.steps:
            return
            
        console.print("\n[bold blue]🔍 Research Steps Summary:[/bold blue]")
        
        summary_table = Table(show_header=True, header_style="bold cyan")
        summary_table.add_column("Step", style="white", width=20)
        summary_table.add_column("Status", width=10)  
        summary_table.add_column("Time", style="cyan", width=8)
        summary_table.add_column("Results", style="dim", width=40)
        
        for step in self.steps:
            step_name = step["name"]
            
            # Status emoji
            if step["status"] == "completed":
                status = "[green]✅ Done[/green]"
            elif step["status"] == "failed":
                status = "[red]❌ Failed[/red]"
            else:
                status = "[yellow]🔄 Running[/yellow]"
            
            # Timing
            timing = f"{step_timings.get(step_name, 0):.2f}s"
            
            # Results summary
            results = step_results.get(step_name, {})
            result_text = ""
            
            if step_name == "retrieve_memory":
                if results.get("memory_found"):
                    result_text = f"Found {results.get('similar_messages', 0)} similar, {results.get('recent_messages', 0)} recent"
                else:
                    result_text = "No memory context"
                    
            elif step_name == "retrieve_documents":
                count = results.get("documents_found", 0)
                result_text = f"Found {count} documents"
                if count > 0 and results.get("document_titles"):
                    titles = results["document_titles"][:2]
                    result_text += f": {', '.join(titles)}"
                    if len(results["document_titles"]) > 2:
                        result_text += "..."
                        
            elif step_name == "search_knowledge_graph":
                count = results.get("kg_results_found", 0)
                result_text = f"Found {count} KG entities"
                if count > 0 and results.get("kg_entities"):
                    entities = results["kg_entities"][:2]
                    result_text += f": {', '.join(entities)}"
                    
            elif step_name == "web_search":
                count = results.get("web_results_found", 0)
                result_text = f"Found {count} web results"
                
            elif step_name == "synthesize_answer":
                length = results.get("answer_length", 0)
                sources = results.get("sources_used", [])
                result_text = f"Generated {length} chars, used: {', '.join(sources)}"
            
            summary_table.add_row(step_name, status, timing, result_text)
        
        console.print(summary_table)


step_tracker = StepTracker()


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
    
    console.print(table)


def display_ref_table(references):
    """Display document references with clickable links."""
    if not references:
        return
    
    console.print("\n[bold blue]📚 Document References:[/bold blue]")
    
    ref_table = Table(show_header=True, header_style="bold cyan", border_style="blue")
    ref_table.add_column("#", style="cyan", width=4)
    ref_table.add_column("Title", style="white", width=25)
    ref_table.add_column("Source", style="dim", width=15)
    ref_table.add_column("Link", style="blue", width=45)
    
    for ref in references:
        # Create clickable link
        full_url = ref['url']
        
        # For display, truncate URL if too long but keep it clickable
        if len(full_url) > 42:
            display_url = full_url[:39] + "..."
        else:
            display_url = full_url
        
        # Create clickable link using Rich markup
        clickable_link = f"[link={full_url}]{display_url}[/link]"
        
        ref_table.add_row(
            str(ref['number']),
            ref['title'],
            ref['source'],
            clickable_link
        )
    
    console.print(ref_table)
    
def display_document_references(references: list):
    """Display document references with clickable links."""
    if not references:
        return
    # Also display the full URLs separately for easy copying
    console.print("\n[dim]References:[/dim]")
    for ref in references:
        console.print(f"[dim]{ref['number']}. {ref['url']}[/dim]")


async def chat_loop():
    """Main chat loop with memory support."""
    display_welcome()
    
    # Get user credentials for memory
    user_id = input("Enter your user ID (default: test_user): ").strip() or "test_user"
    profile_id = input("Enter your profile ID (default: default_profile): ").strip() or "default_profile"
    
    console.print(f"\n[green]👤 User: {user_id} | Profile: {profile_id}[/green]")
    
    # Initialize agent with memory enabled and step tracking
    agent = ResearchAgent(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_documents=5,
        max_web_results=3,
        enable_kg=False,  # Disabled for stability
        enable_web_search=True,
        enable_memory=True,  # Enable chat history
        step_callback=step_tracker.step_callback  # Enable step tracking
    )
    
    console.print("[green]✓ Research Agent ready with memory![/green]\n")
    
    # Session ID will be managed automatically by the agent
    session_id = None
    
    async with agent:
        while True:
            try:
                # Get user input
                query = input("\n💬 Ask me anything: ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Closing session and saying goodbye... 👋[/yellow]")
                    # Close the current session before exiting
                    await agent.close_current_session(user_id, profile_id, session_id)
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    break
                
                elif query.lower() == 'help':
                    console.print(Panel(
                        "[bold]Available Commands:[/bold]\n\n"
                        "• [cyan]help[/cyan] - Show this help\n"
                        "• [cyan]quit/exit[/cyan] - Exit chat\n\n"
                        "[bold]Memory Features:[/bold]\n"
                        "• Remembers your conversation history\n"
                        "• References past discussions\n"
                        "• Maintains context across sessions\n\n"
                        "[bold]Step Tracking:[/bold]\n"
                        "• Real-time step progress display\n"
                        "• Detailed timing and results\n"
                        "• Complete workflow visibility\n\n"
                        "[bold]Example Questions:[/bold]\n"
                        "• What is LangGraph?\n"
                        "• Latest AI developments\n"
                        "• How does vector search work?",
                        title="Help",
                        border_style="blue"
                    ))
                    continue
                
                # Process research query with memory and step tracking
                console.print(f"\n[yellow]🔍 Researching: {query}[/yellow]")
                console.print("[dim]Following research steps...[/dim]")
                
                # Clear previous step tracking
                step_tracker.steps = []
                step_tracker.step_details = {}
                
                result = await agent.research(
                    query=query,
                    user_id=user_id,
                    profile_id=profile_id,
                    session_id=session_id
                )
                
                # Display answer with streaming effect
                console.print("\n[bold green]📝 Answer:[/bold green]")
                stream_text(result["answer"])
                
                # Display document references if available
                references = result["metadata"].get('document_references', [])
                if references:
                    display_document_references(references)
                
                # Display memory context if available
                memory_context = result.get("memory_context", {})
                if memory_context.get("similar_messages"):
                    console.print("\n[bold blue]🧠 Related from memory:[/bold blue]")
                    for msg in memory_context["similar_messages"][:2]:
                        truncated = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                        console.print(f"[dim]• {truncated} (similarity: {msg['similarity']:.2f})[/dim]")
                
                # Display step summary
                # step_tracker.display_summary(
                #     result.get("step_timings", {}), 
                #     result.get("step_results", {})
                # )
                
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
                enable_web_search=True,
                enable_memory=False  # Disable memory for single queries
            )
            
            console.print(f"[yellow]🔍 Researching: {query}[/yellow]")
            
            async with agent:
                result = await agent.research(query)
                
                console.print("\n[bold green]Answer:[/bold green]")
                stream_text(result["answer"])
                
                # Display document references if available
                references = result["metadata"].get('document_references', [])
                if references:
                    display_document_references(references)
                
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