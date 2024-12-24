def verbosity_to_loguru_level(verbose: int) -> str:
    """Map verbosity level (0-4) to loguru level.

    Args:
        verbose: Verbosity level
            0: Minimal output (only errors)
            1: Show warnings and errors
            2: Show main progress (default)
            3: Show detailed progress
            4: Show everything

    Returns:
        Corresponding loguru level
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
