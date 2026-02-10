"""Logging configuration for BarScan."""

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

# Create a package-level logger
logger = logging.getLogger("barscan")


def setup_logging(level: LogLevel = "WARNING", verbose: bool = False) -> None:
    """Configure logging for the barscan package.

    Args:
        level: Base logging level
        verbose: If True, sets level to DEBUG
    """
    if verbose:
        level = "DEBUG"

    # Configure the barscan logger
    logger.setLevel(getattr(logging, level))

    # Only add handler if none exist (avoid duplicate handlers)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(getattr(logging, level))

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module.

    Args:
        name: Module name (e.g., "genius.client")

    Returns:
        Logger instance
    """
    return logging.getLogger(f"barscan.{name}")
