"""Checkpointing functionality for solvers."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union

import orbax.checkpoint as checkpoint
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from mdpax.core.solver import Solver


class CheckpointMixin(ABC):
    """Mixin to add checkpointing capabilities to a solver.

    This mixin class provides functionality for saving and restoring solver state during
    training. It uses Orbax for efficient checkpointing with optional asynchronous saving.
    Checkpoints include both the solver state and configuration, allowing for complete
    reconstruction of the solver.

    Note:
        Saving and restoring checkpoints relies on the Problem and Solver each having a config
        attribute and checkpointing will not be enabled if either does not have a config
        attribute. All of the example Problems and Solvers have configs but this is not a
        requirement because, it requires them to be defined in a module that is importable by Hydra.

    Attributes:
        checkpoint_dir (Path): Directory where checkpoints are stored.
        checkpoint_frequency (int): Number of iterations between checkpoints, 0 to disable.
        max_checkpoints (int): Maximum number of checkpoints to retain.
        enable_async_checkpointing (bool): Whether async checkpointing is enabled.
        checkpoint_manager (checkpoint.CheckpointManager): Orbax checkpoint manager instance.

    Required Protected Methods:
        _restore_state_from_checkpoint(state: Dict[str, Any]) -> None:
            Restore solver state from a checkpoint state dictionary.
        _get_solver_config() -> Dict[str, Any]:
            Get solver configuration for reconstruction.

    Public Interface:
        setup_checkpointing(): Configure checkpointing behavior
        save(step: int): Save current state
        restore(checkpoint_dir: str | Path, step: Optional[int] = None) -> Solver:
            Restore solver from checkpoint

    Example:
        >>> solver = MySolver(problem,
        ...     checkpoint_dir="checkpoints/my_problem",
        ...     checkpoint_frequency=100,
        ...     enable_async_checkpointing=True
        ... )
        >>> # During training, solver.save(step=iteration) is called periodically
        >>> # Later, restore the solver
        >>> solver = MySolver.restore("checkpoint/my_problem")
    """

    def setup_checkpointing(
        self,
        checkpoint_dir: Optional[Union[str, Path]] = None,
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

        Raises:
            ValueError: If checkpoint_frequency or max_checkpoints is negative.
        """
        # Validation
        if checkpoint_frequency < 0:
            raise ValueError("checkpoint_frequency must be non-negative")
        if max_checkpoints < 0:
            raise ValueError("max_checkpoints must be non-negative")

        # Ensure both problem and solver have config attributes
        if not hasattr(self.problem, "config"):
            logger.warning(
                "Problem does not have a config attribute. "
                "Checkpointing requires a config for complete state restoration. "
                "Disabling checkpointing."
            )
            self.checkpoint_frequency = 0
            return
        elif not hasattr(self, "config"):
            logger.warning(
                "Solver does not have a config attribute. "
                "Checkpointing requires a config for complete state restoration. "
                "Disabling checkpointing."
            )
            self.checkpoint_frequency = 0
            return

        # Store basic settings
        self.checkpoint_frequency = checkpoint_frequency
        self.max_checkpoints = max_checkpoints
        self.enable_async_checkpointing = enable_async_checkpointing
        self.checkpoint_manager = None

        # Early return if checkpointing not requested
        if checkpoint_frequency == 0:
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

        # Create checkpoint manager and save config
        self.checkpoint_manager = self._create_checkpoint_manager(
            self.checkpoint_dir, max_checkpoints, enable_async_checkpointing
        )
        OmegaConf.save(self.config, self.checkpoint_dir / "config.yaml")
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

    @classmethod
    def _create_checkpoint_manager(
        cls,
        checkpoint_dir: Union[str, Path],
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
        step: Optional[int] = None,
        new_checkpoint_dir: Optional[str | Path] = None,
        checkpoint_frequency: Optional[int] = None,
        max_checkpoints: Optional[int] = None,
        enable_async_checkpointing: Optional[bool] = None,
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

        Raises:
            ValueError: If no checkpoints are found in the directory.
        """
        # Initialize checkpoint manager
        checkpoint_dir = Path(checkpoint_dir).absolute()

        # Create solver instance with nested problem
        config = OmegaConf.load(checkpoint_dir / "config.yaml")

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
    def _restore_state_from_checkpoint(self, state: Dict[str, Any]) -> None:
        """Restore solver state from checkpoint.

        Args:
            state: Dictionary containing solver state from checkpoint.
        """
        pass
