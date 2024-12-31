"""Logging utilities for mdpax."""

from typing import Literal

LoguruLevel = Literal["ERROR", "WARNING", "INFO", "DEBUG", "TRACE"]


def verbosity_to_loguru_level(verbose: int) -> LoguruLevel:
    """Map verbosity level to loguru log level.

    Converts an integer verbosity level (0-4) to the corresponding loguru log level.
    This allows users to control logging verbosity using a simple integer scale while
    maintaining consistent logging behavior across the package.

    Args:
        verbose: Verbosity level with the following meanings:
            0: ERROR - Minimal output (only errors)
            1: WARNING - Show warnings and errors
            2: INFO - Show main progress (default)
            3: DEBUG - Show detailed progress
            4: TRACE - Show everything

    Returns:
        The corresponding loguru log level.

    Raises:
        TypeError: If verbose is not an integer.
        ValueError: If verbose is not between 0 and 4.

    Example:
        >>> level = verbosity_to_loguru_level(2)
        >>> print(level)
        'INFO'
    """
    if not isinstance(verbose, int):
        raise TypeError("Verbosity must be an integer")
    if verbose < 0 or verbose > 4:
        raise ValueError("Verbosity must be between 0 and 4")

    return {
        0: "ERROR",  # Only show errors
        1: "WARNING",  # Show warnings and errors
        2: "INFO",  # Show main progress (default)
        3: "DEBUG",  # Show detailed progress
        4: "TRACE",  # Show everything
    }[verbose]
