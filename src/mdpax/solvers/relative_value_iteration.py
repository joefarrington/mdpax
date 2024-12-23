"""Relative value iteration solver for average reward MDPs."""

from pathlib import Path
from typing import Optional, Tuple, Union

import chex
import jax.numpy as jnp
from hydra.conf import dataclass
from loguru import logger

from mdpax.core.problem import Problem
from mdpax.core.solver import SolverInfo, SolverState
from mdpax.solvers.value_iteration import ValueIteration, ValueIterationConfig


@dataclass
class RelativeValueIterationConfig(ValueIterationConfig):
    """Configuration for the Relative Value Iteration solver.

    This solver is designed for average reward problems and extends the base
    value iteration solver. It forces gamma=1.0 as required for average reward
    optimization.
    """

    _target_: str = "mdpax.solvers.relative_value_iteration.RelativeValueIteration"
    gamma: float = 1.0  # Must be 1.0 for average reward problems


@chex.dataclass(frozen=True)
class RelativeValueIterationInfo(SolverInfo):
    """Runtime information for relative value iteration."""

    gain: float


@chex.dataclass(frozen=True)
class RelativeValueIterationState(SolverState):
    """Runtime state for relative value iteration."""

    info: RelativeValueIterationInfo


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
        verbose: int = 2,
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
            verbose=verbose,
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
            if (
                self.is_checkpointing_enabled
                and self.iteration % self.checkpoint_frequency == 0
            ):
                self.save(self.iteration)

        if conv >= self.epsilon:
            logger.info("Maximum iterations reached")

        if self.is_checkpointing_enabled:
            self.save(self.iteration)

        # Extract policy if converged or on final iteration
        logger.info("Extracting policy")
        self.policy = self._extract_policy(self.values)
        logger.info("Policy extracted")

        logger.info("Value iteration completed")
        return self.solver_state

    def _get_solver_config(self) -> RelativeValueIterationConfig:
        """Get solver configuration for reconstruction."""
        return RelativeValueIterationConfig(
            problem=self.problem.get_problem_config(),
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            batch_size=self.batch_size,
            jax_double_precision=self.jax_double_precision,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_frequency=self.checkpoint_frequency,
            max_checkpoints=self.max_checkpoints,
            enable_async_checkpointing=self.enable_async_checkpointing,
        )

    @property
    def solver_state(self) -> RelativeValueIterationState:
        """Get solver state for checkpointing."""
        return RelativeValueIterationState(
            values=self.values,
            policy=self.policy,
            info=RelativeValueIterationInfo(iteration=self.iteration, gain=self.gain),
        )

    def _restore_state_from_checkpoint(self, state: dict) -> None:
        """Restore solver state from checkpoint."""
        solver_state = state["state"]
        self.values = solver_state.values
        self.policy = solver_state.policy
        self.iteration = solver_state.info.iteration
        self.gain = solver_state.info.gain
