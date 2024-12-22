"""Relative value iteration solver for average reward MDPs."""

from pathlib import Path
from typing import Optional, Tuple, Union

import jax.numpy as jnp
from loguru import logger

from mdpax.core.problem import Problem
from mdpax.solvers.value_iteration import ValueIteration


class RelativeValueIteration(ValueIteration):
    """Relative value iteration solver for average reward MDPs.

    This solver extends standard value iteration to handle average reward MDPs by:
    1. Using gamma=1.0 (no discounting)
    2. Tracking and subtracting a gain term to handle unbounded values
    3. Using span (max - min value difference) for convergence

    The solver maintains the same batched processing and multi-device support
    as the parent ValueIteration class.
    """

    def __init__(
        self,
        problem: Problem,
        max_iter: int = 1000,
        epsilon: float = 1e-3,
        batch_size: int = 1024,
        jax_double_precision: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_frequency: int = 0,
        max_checkpoints: int = 1,
        enable_async_checkpointing: bool = True,
    ):
        """Initialize solver.

        Args same as ValueIteration except gamma (fixed at 1.0 for average reward).
        """
        # Initialize with gamma=1 since this is average reward case
        super().__init__(
            problem,
            gamma=1.0,  # Fixed for average reward case
            max_iter=max_iter,
            epsilon=epsilon,
            batch_size=batch_size,
            jax_double_precision=jax_double_precision,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=checkpoint_frequency,
            max_checkpoints=max_checkpoints,
            enable_async_checkpointing=enable_async_checkpointing,
        )
        self.gain = 0.0  # Initialize gain term for average reward

    def _iteration_step(self) -> Tuple[jnp.ndarray, float]:
        """Run one iteration and compute span for convergence.

        Returns:
            Tuple of (new_values, span) where span is used for convergence
        """
        # Get new values using parent's batch processing
        new_values, _ = super()._iteration_step()

        # Calculate value differences
        value_diffs = new_values - self.values

        # Update gain using maximum difference
        max_diff = jnp.max(value_diffs)
        self.gain = max_diff  # Set gain to maximum difference

        # Subtract gain from new values
        new_values = new_values - self.gain

        # Compute span for convergence check
        span = max_diff - jnp.min(value_diffs)

        return new_values, span

    def solve(self) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """Run relative value iteration to convergence.

        Returns:
            Tuple of (optimal values, optimal policy, gain)
        """
        while self.iteration < self.max_iter:
            # Perform iteration step
            new_values, conv = self._iteration_step()

            # Update values and iteration count
            self.values = new_values
            self.iteration += 1
            logger.info(
                f"Iteration {self.iteration}: span: {conv:.4f}, gain: {self.gain:.4f}"
            )

            # Check convergence
            if conv < self.epsilon:
                logger.info(
                    f"Convergence threshold reached at iteration {self.iteration}"
                )
                break

            # Save checkpoint if enabled
            if self.is_checkpointing_enabled:
                self.save_checkpoint(self.iteration)

        if conv >= self.epsilon:
            logger.info("Maximum iterations reached")

        # Extract policy if converged or on final iteration
        logger.info("Extracting policy")
        self.policy = self._extract_policy(self.values)
        logger.info("Policy extracted")

        logger.info("Value iteration completed")
        return self.values, self.policy, self.gain

    def _get_checkpoint_state(self) -> dict:
        """Get solver state for checkpointing."""
        cp_state = super()._get_checkpoint_state()
        cp_state["gain"] = self.gain
        return cp_state

    def _restore_from_checkpoint(self, cp_state: dict) -> None:
        """Restore solver state from checkpoint."""
        super()._restore_from_checkpoint(cp_state)
        self.gain = cp_state["gain"]
