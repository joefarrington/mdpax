"""Checkpointing functionality for solvers."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import orbax.checkpoint as checkpoint
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from mdpax.core.solver import Solver


class CheckpointMixin(ABC):
    """Mixin to add checkpointing capabilities to a solver.

    Quick Start:
        - Working with built-in problems? Use restore() to load checkpoints
        - Working in a notebook with a custom defined problem? Use load_checkpoint()
        - Not sure? The system will automatically detect which mode to use and log accordingly

    The checkpointing system uses Hydra's structured configs, which are Python dataclasses
    that contain all information needed to instantiate objects. These configs exist for all
    example Problems and Solvers in mdpax (see Forest, DeMoorSingleProduct, etc.).

    For custom problems, there are two options:

    1. Full Reconstruction:
       - Define your problem in a module (not a notebook)
       - Create a Hydra config class for your problem (see Forest for a simple example)
       - Allows automatic reconstruction of both problem and solver

    2. Manual Reconstruction:
       - Use when working in notebooks or without configs
       - Requires manually reconstructing problem and solver before loading state
       - More flexible but less automated

    Required Implementation:
        _restore_state_from_checkpoint(state: Dict[str, Any]) -> None:
            Restore solver state from a checkpoint state dictionary.

    Attributes:
        checkpoint_dir (Path): Directory where checkpoints are stored.
        checkpoint_frequency (int): Number of iterations between checkpoints, 0 to disable.
        max_checkpoints (int): Maximum number of checkpoints to retain.
        enable_async_checkpointing (bool): Whether async checkpointing is enabled.
        checkpoint_manager (checkpoint.CheckpointManager): Orbax checkpoint manager instance.

    Examples:
        >>> # Full Reconstruction with built-in problem
        >>> from mdpax.problems import Forest  # Problem with Hydra config
        >>> from mdpax.solvers import ValueIteration  # Solver with Hydra config
        >>>
        >>> problem = Forest(S=4)
        >>> solver = ValueIteration(
        ...     problem=problem,
        ...     checkpoint_dir="checkpoints/run1",
        ...     checkpoint_frequency=5
        ... )
        >>> solver.solve(max_iterations=100)
        >>>
        >>> # Later, full reconstruction:
        >>> solver = ValueIteration.restore("checkpoints/run1")  # Recreates everything

        >>> # Manual Reconstruction (e.g., custom problem in notebook)
        >>> class MyProblem:  # Custom problem without config
        ...     def __init__(self, size):
        ...         self.size = size
        ...
        >>> problem = MyProblem(size=4)
        >>> solver = ValueIteration(
        ...     problem=problem,
        ...     checkpoint_dir="checkpoints/run1",
        ...     checkpoint_frequency=5
        ... )
        >>> solver.solve(max_iterations=100)
        >>>
        >>> # Later, manual reconstruction:
        >>> problem = MyProblem(size=4)  # Must recreate problem
        >>> solver = ValueIteration(
        ...     problem=problem,
        ...     checkpoint_dir="checkpoints/run2",  # New save location
        ... )
        >>> solver.load_checkpoint("checkpoints/run1")  # Load from original location
        >>> solver.solve(max_iterations=50)  # New checkpoints go to run2
    """

    def _setup_checkpointing(
        self,
        checkpoint_dir: str | Path | None = None,
        checkpoint_frequency: int = 0,
        max_checkpoints: int = 1,
        enable_async_checkpointing: bool = True,
    ) -> None:
        """Configure checkpointing behavior for the solver.

        Args:
            checkpoint_dir: Directory for storing checkpoints. If None, creates a
                timestamped directory under 'checkpoints/{problem_name}/'.
            checkpoint_frequency: How often to save checkpoints in iterations.
                Set to 0 to disable checkpointing.
            max_checkpoints: Maximum number of checkpoints to retain. Older
                checkpoints are automatically removed.
            enable_async_checkpointing: Whether to use asynchronous checkpointing
                for better performance.
        """
        # Validation
        if checkpoint_frequency < 0:
            raise ValueError("checkpoint_frequency must be non-negative")
        if max_checkpoints < 0:
            raise ValueError("max_checkpoints must be non-negative")

        # Store basic settings
        self.checkpoint_frequency = checkpoint_frequency
        self.max_checkpoints = max_checkpoints
        self.enable_async_checkpointing = enable_async_checkpointing
        self.checkpoint_manager = None

        # Early return if checkpointing not requested
        if self.checkpoint_frequency == 0:
            logger.info("Checkpointing not enabled")
            return

        # Setup checkpoint directory
        if checkpoint_dir is None:
            from datetime import datetime

            current_datetime = datetime.now().strftime("%Y%m%d/%H:%M:%S")
            checkpoint_dir = Path(
                f"checkpoints/{self.problem.name}/{current_datetime}/"
            )
        self.checkpoint_dir = Path(checkpoint_dir).absolute()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Create checkpoint manager and save config
        self.checkpoint_manager = self._create_checkpoint_manager(
            self.checkpoint_dir, max_checkpoints, enable_async_checkpointing
        )

        if self.has_full_config:
            self._save_solver_config()
            logger.info(
                "Full checkpointing enabled with problem and solver reconstruction"
            )
        else:
            logger.info(
                "Lightweight checkpointing enabled - problem and solver must be "
                "reconstructed manually"
            )

        logger.info(
            f"Saving checkpoints every {self.checkpoint_frequency} "
            f"iteration(s) to {self.checkpoint_dir}"
        )

    @property
    def is_checkpointing_enabled(self) -> bool:
        """Check if checkpointing is enabled.

        Returns:
            True if checkpointing is properly configured and enabled, False otherwise.
        """
        return (
            hasattr(self, "checkpoint_frequency")
            and self.checkpoint_frequency > 0
            and hasattr(self, "checkpoint_manager")
        )

    @property
    def has_full_config(self) -> bool:
        """Check if solver and problem have complete configs for reconstruction.

        Checks that both solver and problem have configs with _target_ properties,
        which are required for Hydra to reconstruct the objects.

        Returns:
            True if both solver and problem have complete configs, False otherwise.
        """
        return (
            hasattr(self.problem, "config")
            and self.problem.config is not None
            and hasattr(self, "config")
            and self.config is not None
            and hasattr(self.problem.config, "_target_")
            and self.problem.config._target_ is not None
            and hasattr(self.config, "_target_")
            and self.config._target_ is not None
        )

    def load_checkpoint(
        self, checkpoint_dir: str | Path, step: int | None = None
    ) -> None:
        """Load solver state from checkpoint.

        Must be called on an already constructed solver instance with problem.
        Will load state from checkpoint_dir, but continue saving new checkpoints
        to the directory specified during construction (self.checkpoint_dir).

        Args:
            checkpoint_dir: Directory containing checkpoint to load from
            step: Specific step to load. If None, loads the latest checkpoint.
        Returns:
            None
        """
        checkpoint_dir = Path(checkpoint_dir).absolute()
        load_manager = self._create_checkpoint_manager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=1,  # Only need to keep one when loading
            enable_async_checkpointing=True,
        )

        template_cp_state = self.solver_state
        step = step or load_manager.latest_step()
        if step is None:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")

        cp_state = load_manager.restore(
            step,
            args=checkpoint.args.StandardRestore(template_cp_state),
        )

        self._restore_state_from_checkpoint(cp_state)

    def _save_solver_config(self) -> None:
        """Save the solver config to the checkpoint directory."""
        OmegaConf.save(self.config, self.checkpoint_dir / "config.yaml")

    @classmethod
    def _create_checkpoint_manager(
        cls,
        checkpoint_dir: str | Path,
        max_checkpoints: int,
        enable_async_checkpointing: bool,
    ) -> checkpoint.CheckpointManager:
        """Create an Orbax checkpoint manager.

        Args:
            checkpoint_dir: Directory for storing checkpoints.
            max_checkpoints: Maximum number of checkpoints to retain.
            enable_async_checkpointing: Whether to use async checkpointing.

        Returns:
            Configured Orbax checkpoint manager.
        """
        # Configure Orbax
        options = checkpoint.CheckpointManagerOptions(
            max_to_keep=max_checkpoints,
            create=True,
            enable_async_checkpointing=enable_async_checkpointing,
        )

        return checkpoint.CheckpointManager(
            checkpoint_dir,
            options=options,
        )

    @contextmanager
    def _checkpoint_operation(self):
        """Context manager for checkpoint operations.

        Ensures async operations complete before exiting context.
        """
        try:
            yield
        finally:
            if self.enable_async_checkpointing:
                self.checkpoint_manager.wait_until_finished()

    def save(self, step: int) -> None:
        """Save current solver state to checkpoint.

        Args:
            step: Current iteration/step number to associate with the checkpoint.

        Returns:
            None
        """
        if not self.is_checkpointing_enabled:
            return

        # Get state to checkpoint
        cp_state = self.solver_state

        # Save checkpoint
        self.checkpoint_manager.save(step, args=checkpoint.args.StandardSave(cp_state))

        status = "queued" if self.enable_async_checkpointing else "saved"
        logger.debug(f"Checkpoint {status} for iteration {step}")

    @classmethod
    def restore(
        cls,
        checkpoint_dir: str | Path,
        step: int | None = None,
        new_checkpoint_dir: str | Path | None = None,
        checkpoint_frequency: int | None = None,
        max_checkpoints: int | None = None,
        enable_async_checkpointing: bool | None = None,
    ) -> Solver:
        """Load solver from checkpoint.

        This class method reconstructs a solver instance from a checkpoint,
        using the stored config to recreate both the problem and solver
        with the correct parameters.

        Args:
            checkpoint_dir: Directory containing checkpoints.
            step: Specific step to load. If None, loads the latest checkpoint.
            new_checkpoint_dir: Optional new directory for future checkpoints.
                Useful when restoring to a different location.
            checkpoint_frequency: Optional new checkpoint frequency.
            max_checkpoints: Optional new maximum number of checkpoints.
            enable_async_checkpointing: Optional new async checkpointing setting.

        Returns:
            Reconstructed solver instance with restored state.
        """
        # Initialize checkpoint manager
        checkpoint_dir = Path(checkpoint_dir).absolute()
        config_path = checkpoint_dir / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"No config file found in {checkpoint_dir}. "
                "If you're working with a problem without configs (e.g., in a notebook), "
                "you need to:\n"
                "1. Create your problem instance\n"
                "2. Create your solver instance with that problem\n"
                "3. Use solver.load_checkpoint() instead of the class restore() method"
            )

        # Create solver instance with nested problem
        config = OmegaConf.load(config_path)

        # Update config with new values (only allow new values if provided)
        if new_checkpoint_dir is not None:
            new_checkpoint_dir = Path(new_checkpoint_dir).absolute()
            config.checkpoint_dir = new_checkpoint_dir

        if checkpoint_frequency is not None:
            config.checkpoint_frequency = checkpoint_frequency

        if max_checkpoints is not None:
            config.max_checkpoints = max_checkpoints

        if enable_async_checkpointing is not None:
            config.enable_async_checkpointing = enable_async_checkpointing

        solver = instantiate(config)

        template_cp_state = solver.solver_state
        manager = cls._create_checkpoint_manager(checkpoint_dir, 1, True)

        # Get step to restore
        step = step or manager.latest_step()
        if step is None:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")

        # Restore state
        cp_state = manager.restore(
            step,
            args=checkpoint.args.StandardRestore(template_cp_state),
        )

        # Restore runtime state
        solver._restore_state_from_checkpoint(cp_state)

        return solver

    @abstractmethod
    def _restore_state_from_checkpoint(self, state: dict[str, Any]) -> None:
        """Restore solver state from checkpoint.

        Args:
            state: Dictionary containing solver state from checkpoint.
        """
        pass
