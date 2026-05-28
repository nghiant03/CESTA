"""Centralized logging configuration using loguru.

This module provides a pre-configured logger for the CESTA project.
"""

import sys

from loguru import logger

# Remove default handler
logger.remove()

# Add custom handler with pretty formatting
DEFAULT_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
VERBOSE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
COMPACT_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<level>{message}</level>"
)

logger.add(
    sys.stderr,
    format=DEFAULT_FORMAT,
    level="INFO",
    colorize=True,
)


def configure_logging(level: str = "INFO", verbose: bool = False) -> None:
    """Configure logging level.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        verbose: If True, show full module path in logs.
    """
    logger.remove()

    fmt = VERBOSE_FORMAT if verbose else COMPACT_FORMAT

    logger.add(
        sys.stderr,
        format=fmt,
        level=level.upper(),
        colorize=True,
    )
