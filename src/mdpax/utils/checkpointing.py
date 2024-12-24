"""Checkpointing functionality for solvers."""

from abc import ABC
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import orbax.checkpoint as checkpoint
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from mdpax.core.solver import Solver


class CheckpointMixin(ABC):
    """Mixin to add checkpointing capabilities to a solver.

    Provides functionality for:
    - Periodic checkpointing of solver state
    - Optional async saving for better performance

    Required Protected Methods:
    -------------------------
    _restore_from_checkpoint(cp_state: dict) -> None:
        Restore solver state from a checkpoint state dict.

    Public Interface:
    ---------------
    setup_checkpointing(): Configure checkpointing behavior
    save_checkpoint(): Save current state
    restore_latest_checkpoint(): Restore most recent checkpoint
    """

    def setup_checkpointing(
        self,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_frequency: int = 0,
        max_checkpoints: int = 1,
        enable_async_checkpointing: bool = True,
    ) -> None:
        """Setup checkpointing for the solver.

        Args:
            checkpoint_dir: Directory for checkpoints
            checkpoint_frequency: How often to save checkpoints (iterations)
            max_checkpoints: Maximum number of checkpoints to keep
            enable_async: Whether to use async checkpointing
        """
        # Validation
        if checkpoint_frequency < 0:
            raise ValueError("checkpoint_frequency must be non-negative")
        if max_checkpoints < 0:
            raise ValueError("max_checkpoints must be non-negative")

        if checkpoint_dir is None:
            from datetime import datetime

            current_datetime = datetime.now().strftime("%Y%m%d/%H:%M:%S")
            checkpoint_dir = Path(
                f"checkpoints/{self.problem.name}/{current_datetime}/"
            )

        self.checkpoint_dir = Path(checkpoint_dir).absolute()
        self.checkpoint_frequency = checkpoint_frequency
        self.max_checkpoints = max_checkpoints
        self.enable_async_checkpointing = enable_async_checkpointing

        if checkpoint_frequency > 0:
            self.checkpoint_manager = self._create_checkpoint_manager(
                self.checkpoint_dir, max_checkpoints, enable_async_checkpointing
            )
            OmegaConf.save(
                self._get_solver_config(), self.checkpoint_dir / "config.yaml"
            )
            logger.info(
                f"Saving checkpoints every {self.checkpoint_frequency}\
                iteration(s) to {self.checkpoint_dir}"
            )
        else:
            self.checkpoint_manager = None
            logger.info("Checkpointing disabled")

    @property
    def is_checkpointing_enabled(self) -> bool:
        """Whether checkpointing is enabled."""
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
        """Context manager for checkpoint operations."""
        try:
            yield
        finally:
            if self.enable_async_checkpointing:
                self.checkpoint_manager.wait_until_finished()

    def save(self, step: int) -> None:
        """Save solver state to checkpoint.

        Args:
            step: Current iteration/step number
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
    ) -> Solver:
        """Load solver from checkpoint.

        This class method reconstructs a solver instance from a checkpoint,
        using the stored config to recreate both the problem and solver
        with the correct parameters.

        Args:
            checkpoint_dir: Directory containing checkpoints
            step: Specific step to load (defaults to latest)

        Returns:
            Reconstructed solver instance
        """
        # Initialize checkpoint manager
        checkpoint_dir = Path(checkpoint_dir).absolute()

        # Create solver instance with nested problem
        config = OmegaConf.load(checkpoint_dir / "config.yaml")
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

        if new_checkpoint_dir is not None:
            new_checkpoint_dir = Path(new_checkpoint_dir).absolute()
            cp_state["config"]["checkpoint_dir"] = new_checkpoint_dir

        # Restore runtime state
        solver._restore_state_from_checkpoint(cp_state)

        return solver

    def _get_solver_config(self) -> dict:
        """Get solver configuration for reconstruction.

        This should be implemented by solvers to return their Hydra config.
        """
        raise NotImplementedError("Solvers must implement _get_solver_config")

    def _restore_state_from_checkpoint(self, state: dict) -> None:
        """Restore solver state from checkpoint."""
        raise NotImplementedError("Solvers must implement _restore_from_checkpoint")
