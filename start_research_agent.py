#!/usr/bin/env python3
"""Entry point for the Research Agent Chat."""

def main():
    """Main entry point for the research chat application."""
    from app.agents.research_agent.test.chat import main as chat_main
    chat_main()

if __name__ == "__main__":
    main()