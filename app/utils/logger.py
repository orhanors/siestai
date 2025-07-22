"""
Rich logger configuration for the Siestai application.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
from rich.theme import Theme
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm
from rich.align import Align

# Install rich traceback handler for better error formatting
install(show_locals=True)

# Custom theme for consistent styling
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green",
    "debug": "dim",
    "database": "blue",
    "api": "magenta",
    "intercom": "bright_blue",
    "jira": "bright_yellow",
    "confluence": "bright_magenta"
})

# Create console instance
console = Console(theme=custom_theme, width=120)


class SiestaiLogger:
    """
    Enhanced logger for Siestai application with rich formatting.
    """
    
    def __init__(
        self,
        name: str = "siestai",
        level: str = "INFO",
        log_file: Optional[str] = None,
        enable_console: bool = True
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            enable_console: Whether to enable console output
        """
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers only if this is a new instance
        if not self.logger.handlers:
            # Create formatters
            self.console_formatter = logging.Formatter(
                fmt="%(message)s",
                datefmt="[%X]"
            )
            
            self.file_formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            
            # Add console handler with rich formatting
            if enable_console:
                console_handler = RichHandler(
                    console=console,
                    show_time=True,
                    show_path=False,
                    markup=True,
                    rich_tracebacks=True,
                    tracebacks_show_locals=True
                )
                console_handler.setFormatter(self.console_formatter)
                self.logger.addHandler(console_handler)
            
            # Add file handler if specified
            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(self.file_formatter)
                self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def success(self, message: str):
        """Log success message with green color."""
        self.logger.info(f"[green]✓ {message}[/green]")
    
    def database(self, message: str):
        """Log database operation message."""
        self.logger.info(f"[database]🗄️  {message}[/database]")
    
    def api(self, message: str):
        """Log API operation message."""
        self.logger.info(f"[api]🌐 {message}[/api]")
    
    def intercom(self, message: str):
        """Log Intercom operation message."""
        self.logger.info(f"[intercom]💬 {message}[/intercom]")
    
    def jira(self, message: str):
        """Log Jira operation message."""
        self.logger.info(f"[jira]📋 {message}[/jira]")
    
    def confluence(self, message: str):
        """Log Confluence operation message."""
        self.logger.info(f"[confluence]📚 {message}[/confluence]")
    def document(self, message: str):
        """Log document operation message."""
        self.logger.info(f"[document]📄 {message}[/document]")


class ProgressLogger:
    """
    Progress tracking with rich progress bars.
    """
    
    def __init__(self, description: str = "Processing"):
        self.description = description
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        )
    
    def __enter__(self):
        self.progress.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()
    
    def add_task(self, description: str, total: int = 100) -> int:
        """Add a new task to track."""
        return self.progress.add_task(description, total=total)
    
    def update(self, task_id: int, advance: int = 1):
        """Update task progress."""
        self.progress.update(task_id, advance=advance)
    
    def complete(self, task_id: int):
        """Mark task as complete."""
        self.progress.update(task_id, completed=self.progress.tasks[task_id].total)


class StatusLogger:
    """
    Status display with rich panels and tables.
    """
    
    @staticmethod
    def show_startup_banner():
        """Display application startup banner."""
        banner = Text("🚀 Siestai AI Agent Platform", style="bold blue")
        subtitle = Text("Intelligent document processing and conversation management", style="dim")
        
        panel = Panel(
            Align.center(banner + "\n" + subtitle),
            border_style="blue",
            padding=(1, 2)
        )
        console.print(panel)
    
    @staticmethod
    def show_status_table(status_data: Dict[str, Any]):
        """Display status information in a table."""
        table = Table(title="System Status", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Details", style="white")
        
        for component, data in status_data.items():
            status_icon = "✓" if data.get("status") == "ok" else "✗"
            status_color = "green" if data.get("status") == "ok" else "red"
            table.add_row(
                component,
                f"[{status_color}]{status_icon} {data.get('status', 'unknown')}[/{status_color}]",
                str(data.get("details", ""))
            )
        
        console.print(table)
    
    @staticmethod
    def show_document_stats(stats: Dict[str, Any]):
        """Display document statistics."""
        table = Table(title="Document Statistics", show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in stats.items():
            if isinstance(value, dict):
                value_str = ", ".join([f"{k}: {v}" for k, v in value.items()])
            else:
                value_str = str(value)
            table.add_row(metric.replace("_", " ").title(), value_str)
        
        console.print(table)


# Global logger instances - use singleton pattern to prevent duplicates
_logger_instances = {}

def get_logger(name: str = "siestai") -> SiestaiLogger:
    """Get or create a logger instance with the given name."""
    if name not in _logger_instances:
        _logger_instances[name] = SiestaiLogger(name)
    return _logger_instances[name]

# Convenience functions for common loggers
logger = get_logger("siestai")
db_logger = get_logger("siestai.database")
api_logger = get_logger("siestai.api")
intercom_logger = get_logger("siestai.intercom")
jira_logger = get_logger("siestai.jira")
confluence_logger = get_logger("siestai.confluence")


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True
) -> SiestaiLogger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        enable_console: Whether to enable console output
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"siestai_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Setup main logger
    main_logger = SiestaiLogger(
        name="siestai",
        level=level,
        log_file=str(log_file),
        enable_console=enable_console
    )
    
    # Setup component loggers
    components = ["database", "api", "intercom", "jira", "confluence"]
    for component in components:
        SiestaiLogger(
            name=f"siestai.{component}",
            level=level,
            log_file=str(log_file),
            enable_console=enable_console
        )
    
    return main_logger


def log_function_call(func_name: str, args: tuple = None, kwargs: dict = None):
    """
    Decorator to log function calls.
    
    Args:
        func_name: Function name
        args: Function arguments
        kwargs: Function keyword arguments
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func_name} failed with error: {e}")
                raise
        return wrapper
    return decorator


# Example usage functions
def example_logging():
    """Example of how to use the rich logger."""
    
    # Show startup banner
    StatusLogger.show_startup_banner()
    
    # Setup logging
    setup_logging(level="DEBUG")
    
    # Log different types of messages
    logger.info("Application started")
    logger.success("Database connection established")
    logger.database("Executing query: SELECT * FROM documents")
    logger.api("Making request to Intercom API")
    logger.intercom("Fetched 25 articles from Intercom")
    logger.warning("Rate limit approaching for API calls")
    logger.error("Failed to connect to external service")
    
    # Show status table
    status_data = {
        "Database": {"status": "ok", "details": "Connected to PostgreSQL"},
        "Intercom API": {"status": "ok", "details": "Rate limit: 85/100"},
        "Vector Search": {"status": "ok", "details": "pgvector extension active"},
        "File Storage": {"status": "error", "details": "Permission denied"}
    }
    StatusLogger.show_status_table(status_data)
    
    # Show document stats
    stats = {
        "total_documents": 1250,
        "documents_with_embeddings": 1180,
        "unique_sources": {"intercom": 450, "jira": 300, "confluence": 500},
        "languages": {"en": 1000, "es": 150, "fr": 100}
    }
    StatusLogger.show_document_stats(stats)
    
    # Progress tracking example
    with ProgressLogger("Processing documents") as progress:
        task = progress.add_task("Fetching from Intercom", total=100)
        for i in range(10):
            progress.update(task, advance=10)
            import time
            time.sleep(0.1)


if __name__ == "__main__":
    example_logging() 