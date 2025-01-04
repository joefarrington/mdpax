"""Logging utilities."""

from typing import Literal

import numpy as np

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


def get_convergence_format(epsilon: float, max_decimals: int = 10) -> str:
    """Get format string to show changes above epsilon.

    Creates a format string (e.g. '.4f') that ensures logged values will show
    meaningful changes until convergence is reached. This is useful for logging
    convergence metrics like span or residual in solvers.

    Args:
        epsilon: Convergence threshold
        max_decimals: Maximum number of decimal places to show (default: 10)

    Returns:
        Format string (e.g. '.4f' or '.6f') suitable for f-string formatting

    Example:
        >>> fmt = get_convergence_format(1e-6)
        >>> print(fmt)
        '.7f'
        >>> value = 0.0000123
        >>> print(f"Value: {value:{fmt}}")
        Value: 0.0000123
    """
    if not isinstance(epsilon, float):
        raise TypeError("epsilon must be a float")
    if not isinstance(max_decimals, int):
        raise TypeError("max_decimals must be an integer")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_decimals <= 0:
        raise ValueError("max_decimals must be positive")

    # Get number of decimal places needed to show changes above epsilon
    # Add 1 to ensure we can see changes until below epsilon
    decimal_places = -int(np.floor(np.log10(epsilon))) + 1
    # Cap at max_decimals
    decimal_places = min(decimal_places, max_decimals)

    return f".{decimal_places}f"
