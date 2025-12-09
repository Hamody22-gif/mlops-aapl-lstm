"""
Logger Configuration for Stock ML Project
This module sets up Loguru with:
- Colored console output
- File logging with rotation
- Automatic log cleanup
"""

from loguru import logger
import sys
import os


def setup_logger():
    """
    Configure Loguru for the ML project
    
    Features:
    - Console output with colors and timestamps
    - File output saved to logs/ directory
    - Automatic file rotation (500 MB)
    - 10-day retention of old logs
    - Compression of archived logs
    """
    
    # Remove default logger
    logger.remove()
    
    # Add console output with colors
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Add file output
    logger.add(
        "logs/training_{time:YYYY-MM-DD}.log",
        rotation="500 MB",      # Create new file when size hits 500MB
        retention="10 days",     # Keep logs for 10 days
        compression="zip",       # Compress old logs
        level="DEBUG"            # Save everything to file (including DEBUG messages)
    )
    
    logger.info("Logger configured successfully")
    return logger
