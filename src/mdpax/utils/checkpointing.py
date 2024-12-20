from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import orbax.checkpoint as checkpoint
from loguru import logger


class CheckpointMixin(ABC):
    """Mixin to add checkpointing capabilities to a solver.

    Provides functionality for:
    - Periodic checkpointing of solver state
    - Optional async saving for better performance

    Required Protected Methods:
    -------------------------
    _get_checkpoint_state() -> dict:
        Return a dict containing all state needed to resume solving.
        Must include at minimum:
        - values: Current value function
        - iteration: Current iteration count
        Should include any solver-specific state.

    _restore_from_checkpoint(cp_state: dict) -> None:
        Restore solver state from a checkpoint state dict.
        Must handle all state saved by _get_checkpoint_state.

    Public Interface:
    ---------------
    setup_checkpointing(): Configure checkpointing behavior
    save_checkpoint(): Save current state
    restore_latest_checkpoint(): Restore most recent checkpoint
    """

    def setup_checkpointing(
        self,
        checkpoint_dir: Union[str, Path],
        checkpoint_frequency: int,
        *,  # Force keyword arguments for optional parameters
        max_checkpoints: int = 1,
        enable_async: bool = False,
    ) -> None:
        """Set up checkpointing infrastructure."""
        # Validation
        if checkpoint_frequency < 0:
            raise ValueError("checkpoint_frequency must be non-negative")
        if max_checkpoints < 0:
            raise ValueError("max_checkpoints must be non-negative")

        if checkpoint_dir is None:
            checkpoint_dir = (
                Path(self.problem.name)
                / "checkpoints"
                / datetime.now().strftime("%Y-%m-%d/%H:%M:%S")
            )

        # Setup
        self.checkpoint_dir = Path(checkpoint_dir).absolute()
        self.checkpoint_frequency = checkpoint_frequency
        self.max_checkpoints = max_checkpoints
        self.enable_async = enable_async
        self._setup_checkpoint_manager(
            self.checkpoint_dir, max_checkpoints, enable_async
        )

    def _setup_checkpoint_manager(
        self, checkpoint_dir: Union[str, Path], max_checkpoints: int, enable_async: bool
    ) -> None:

        # Configure Orbax
        options = checkpoint.CheckpointManagerOptions(
            max_to_keep=max_checkpoints,
            create=True,
            enable_async_checkpointing=enable_async,
        )

        self.checkpoint_manager = checkpoint.CheckpointManager(
            checkpoint_dir,
            options=options,
        )

    @abstractmethod
    def _get_checkpoint_state(self) -> dict:
        """Get solver state for checkpointing.

        Returns:
            dict containing all necessary state to resume solving:
            - values: Current value function
            - iteration: Current iteration count
            - Any solver-specific state needed for resuming
        """
        pass

    @abstractmethod
    def _restore_from_checkpoint(self, cp_state: dict) -> None:
        """Restore solver state from checkpoint.

        Args:
            cp_state: State dict from _get_checkpoint_state()
        """
        pass

    @property
    def is_checkpointing_enabled(self) -> bool:
        """Whether checkpointing is enabled and properly configured."""
        return (
            hasattr(self, "checkpoint_frequency")
            and self.checkpoint_frequency > 0
            and hasattr(self, "checkpoint_manager")
        )

    @contextmanager
    def _checkpoint_operation(self):
        """Context manager for checkpoint operations."""
        try:
            yield
        finally:
            if self.enable_async:
                self.checkpoint_manager.wait_until_finished()

    def save_checkpoint(self, step: int) -> None:
        """Save current solver state to checkpoint.

        Args:
            step: Current iteration/step number
        """
        if not self.is_checkpointing_enabled:
            return

        if step % self.checkpoint_frequency == 0:
            with self._checkpoint_operation():
                try:
                    cp_state = self._get_checkpoint_state()
                    self.checkpoint_manager.save(
                        step, args=checkpoint.args.StandardSave(cp_state)
                    )
                    status = "queued" if self.enable_async else "saved"
                    logger.info(f"Checkpoint {status} for step {step}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint at step {step}: {str(e)}")

    def restore_latest_checkpoint(
        self, checkpoint_dir: Optional[Union[str, Path]] = None
    ) -> Optional[int]:
        """Restore the latest checkpoint.

        Args:
            checkpoint_dir: Optional directory to restore from
            (if different from current)

        Returns:
            The step number of the restored checkpoint, or None if no checkpoint found
        """
        if not self.is_checkpointing_enabled:
            logger.warning("Checkpointing is not enabled")
            return None

        if checkpoint_dir is not None:
            restore_checkpoint_dir = Path(checkpoint_dir).absolute()
            self._setup_checkpoint_manager(
                restore_checkpoint_dir,
                max_checkpoints=self.max_checkpoints,
                enable_async=self.enable_async,
            )

        latest_step = None
        with self._checkpoint_operation():
            try:
                latest_step = self.checkpoint_manager.latest_step()
                if latest_step is not None:
                    # Get current state as template
                    template_state = self._get_checkpoint_state()
                    # Restore using template
                    cp_state = self.checkpoint_manager.restore(
                        latest_step,
                        args=checkpoint.args.StandardRestore(template_state),
                    )
                    self._restore_from_checkpoint(cp_state)
                    log_string = " ".join(
                        [
                            "Restored checkpoint from",
                            f"step {latest_step} in {restore_checkpoint_dir}",
                        ]
                    )
                    logger.info(log_string)
            except Exception as e:
                logger.error(f"Failed to restore checkpoint: {str(e)}")
                raise

        # We don't necessarily want to save the checkpoint in the same directory as the
        # one we're restoring from, so we need to reset the checkpoint directory
        self._setup_checkpoint_manager(
            self.checkpoint_dir,
            max_checkpoints=self.max_checkpoints,
            enable_async=self.enable_async,
        )

        return latest_step
