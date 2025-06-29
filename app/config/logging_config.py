"""
Logging configuration for Siestai application.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base directory for logs
LOG_DIR = Path("logs")

# Default logging configuration
DEFAULT_LOGGING_CONFIG = {
    "level": "INFO",
    "console_enabled": True,
    "file_enabled": True,
    "file_rotation": "daily",  # daily, weekly, monthly, or size-based
    "max_file_size": "10MB",
    "backup_count": 7,
    "format": {
        "console": "%(message)s",
        "file": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    },
    "date_format": {
        "console": "[%X]",
        "file": "%Y-%m-%d %H:%M:%S"
    }
}

# Environment-specific configurations
LOGGING_CONFIGS = {
    "development": {
        "level": "DEBUG",
        "console_enabled": True,
        "file_enabled": True,
        "file_rotation": "daily",
        "max_file_size": "5MB",
        "backup_count": 3
    },
    "staging": {
        "level": "INFO",
        "console_enabled": False,
        "file_enabled": True,
        "file_rotation": "daily",
        "max_file_size": "10MB",
        "backup_count": 7
    },
    "production": {
        "level": "WARNING",
        "console_enabled": False,
        "file_enabled": True,
        "file_rotation": "daily",
        "max_file_size": "50MB",
        "backup_count": 30
    }
}

# Component-specific log levels
COMPONENT_LOG_LEVELS = {
    "siestai": "INFO",
    "siestai.database": "INFO",
    "siestai.api": "DEBUG",
    "siestai.intercom": "INFO",
    "siestai.jira": "INFO",
    "siestai.confluence": "INFO",
    "siestai.vector_search": "DEBUG",
    "siestai.embeddings": "DEBUG"
}


def get_logging_config(environment: str = None) -> Dict[str, Any]:
    """
    Get logging configuration for the specified environment.
    
    Args:
        environment: Environment name (development, staging, production)
    
    Returns:
        Logging configuration dictionary
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    # Get base config for environment
    config = DEFAULT_LOGGING_CONFIG.copy()
    if environment in LOGGING_CONFIGS:
        config.update(LOGGING_CONFIGS[environment])
    
    # Override with environment variables if present
    if os.getenv("LOG_LEVEL"):
        config["level"] = os.getenv("LOG_LEVEL")
    
    if os.getenv("LOG_CONSOLE_ENABLED"):
        config["console_enabled"] = os.getenv("LOG_CONSOLE_ENABLED").lower() == "true"
    
    if os.getenv("LOG_FILE_ENABLED"):
        config["file_enabled"] = os.getenv("LOG_FILE_ENABLED").lower() == "true"
    
    return config


def get_log_file_path(environment: str = None) -> Path:
    """
    Get log file path for the specified environment.
    
    Args:
        environment: Environment name
    
    Returns:
        Path to log file
    """
    config = get_logging_config(environment)
    
    # Create logs directory
    LOG_DIR.mkdir(exist_ok=True)
    
    # Generate filename based on environment and date
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")
    env_suffix = f"_{environment}" if environment and environment != "development" else ""
    
    return LOG_DIR / f"siestai{env_suffix}_{date_str}.log"


def get_component_log_level(component: str) -> str:
    """
    Get log level for a specific component.
    
    Args:
        component: Component name
    
    Returns:
        Log level for the component
    """
    return COMPONENT_LOG_LEVELS.get(component, "INFO")


# Log file rotation settings
ROTATION_CONFIGS = {
    "daily": {
        "when": "midnight",
        "interval": 1,
        "backup_count": 7
    },
    "weekly": {
        "when": "W0",
        "interval": 1,
        "backup_count": 4
    },
    "monthly": {
        "when": "M",
        "interval": 1,
        "backup_count": 12
    }
}


def get_rotation_config(rotation_type: str) -> Dict[str, Any]:
    """
    Get rotation configuration for the specified type.
    
    Args:
        rotation_type: Type of rotation (daily, weekly, monthly)
    
    Returns:
        Rotation configuration dictionary
    """
    return ROTATION_CONFIGS.get(rotation_type, ROTATION_CONFIGS["daily"])


# Color themes for different environments
COLOR_THEMES = {
    "development": {
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
    },
    "staging": {
        "info": "blue",
        "warning": "yellow",
        "error": "red",
        "success": "green",
        "debug": "dim",
        "database": "cyan",
        "api": "magenta",
        "intercom": "bright_blue",
        "jira": "bright_yellow",
        "confluence": "bright_magenta"
    },
    "production": {
        "info": "white",
        "warning": "yellow",
        "error": "red",
        "success": "green",
        "debug": "dim",
        "database": "blue",
        "api": "magenta",
        "intercom": "bright_blue",
        "jira": "bright_yellow",
        "confluence": "bright_magenta"
    }
}


def get_color_theme(environment: str = None) -> Dict[str, str]:
    """
    Get color theme for the specified environment.
    
    Args:
        environment: Environment name
    
    Returns:
        Color theme dictionary
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    return COLOR_THEMES.get(environment, COLOR_THEMES["development"]) 