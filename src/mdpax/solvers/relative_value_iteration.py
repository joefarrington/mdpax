"""Relative value iteration solver for average reward MDPs."""

from pathlib import Path

import chex
import jax.numpy as jnp
from hydra.conf import MISSING, dataclass
from loguru import logger

from mdpax.core.problem import Problem, ProblemConfig
from mdpax.core.solver import SolverInfo, SolverState
from mdpax.solvers.value_iteration import ValueIteration
from mdpax.utils.logging import get_convergence_format
from mdpax.utils.types import ValueFunction


@dataclass
class RelativeValueIterationConfig:
    """Configuration for the Relative Value Iteration solver.

    TODO: Need to define fresh because no gamma - need to think
    about best way to deal with this.

    Attributes:
        _target_: Full path to solver class for Hydra instantiation
        gamma: Must be 1.0 for average reward problems
    """

    _target_: str = "mdpax.solvers.relative_value_iteration.RelativeValueIteration"
    problem: ProblemConfig = MISSING

    # Solver parameters
    epsilon: float = 1e-3
    max_batch_size: int = 1024
    jax_double_precision: bool = True
    verbose: int = 2  # Default to INFO level
    checkpoint_dir: str | None = None
    checkpoint_frequency: int = 0
    max_checkpoints: int = 1
    enable_async_checkpointing: bool = True


@chex.dataclass(frozen=True)
class RelativeValueIterationInfo(SolverInfo):
    """Runtime information for relative value iteration.

    Attributes:
        gain: Current gain term for average reward adjustment
    """

    gain: float


@chex.dataclass(frozen=True)
class RelativeValueIterationState(SolverState):
    """Runtime state for relative value iteration.

    Attributes:
        values: Current value function [n_states]
        policy: Current policy [n_states, action_dim]
        info: Solver metadata including gain term
    """

    info: RelativeValueIterationInfo


class RelativeValueIteration(ValueIteration):
    """Relative value iteration solver for average reward MDPs.

    This solver extends standard value iteration to handle average reward MDPs by:
    1. Using gamma=1.0 (no discounting)
    2. Tracking and subtracting a gain term to handle unbounded values
    3. Using span (max - min value difference) for convergence

    Note:
        Supports checkpointing for long-running problems.


     Args:
        problem: MDP problem to solve
        epsilon: Convergence threshold for value changes
        max_batch_size: Maximum states to process in parallel on each device
        jax_double_precision: Whether to use float64 precision
        checkpoint_dir: Directory to store checkpoints
        checkpoint_frequency: How often to save checkpoints (0 to disable)
        max_checkpoints: Maximum number of checkpoints to keep
        enable_async_checkpointing: Whether to save checkpoints asynchronously
        verbose: Logging verbosity level (0-4)
    """

    def __init__(
        self,
        problem: Problem,
        epsilon: float = 1e-3,
        max_batch_size: int = 1024,
        jax_double_precision: bool = True,
        verbose: int = 2,
        checkpoint_dir: str | Path | None = None,
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
            epsilon=epsilon,
            max_batch_size=max_batch_size,
            jax_double_precision=jax_double_precision,
            verbose=verbose,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=checkpoint_frequency,
            max_checkpoints=max_checkpoints,
            enable_async_checkpointing=enable_async_checkpointing,
        )
        # Get convergence format for logging convergence metrics
        self.convergence_format = get_convergence_format(epsilon)
        self.gain = 0.0  # Initialize gain term for average reward

    def _iteration_step(self) -> tuple[ValueFunction, float]:
        """Run one iteration and compute span for convergence.

        Returns:
            Tuple of (new_values, span) where:
                - new_values are the updated state values [n_states]
                - span is max-min value difference for convergence
        """
        # Get new values using parent's batch processing
        new_values, _ = super()._iteration_step()

        # Calculate value differences
        value_diffs = new_values - self.values

        # Update gain using maximum difference
        max_diff = jnp.max(value_diffs)
        self.gain = max_diff

        # Subtract gain from new values
        new_values = new_values - self.gain

        # Compute span for convergence check
        span = max_diff - jnp.min(value_diffs)

        return new_values, span

    def solve(self, max_iterations: int = 2000) -> RelativeValueIterationState:
        """Run relative value iteration.

        Performs synchronous value iteration updates until either:
        1. The span of value differences is below epsilon
        2. The maximum number of iterations is reached

        Args:
            max_iterations: Maximum number of iterations to run

        Returns:
            SolverState containing:
                - Final values [n_states]
                - Optimal policy [n_states, action_dim]
                - Solver info including iteration count and gain term
        """
        for _ in range(max_iterations):
            self.iteration += 1
            new_values, conv = self._iteration_step()
            self.values = new_values

            logger.info(
                f"Iteration {self.iteration}: span: {conv:{self.convergence_format}}, gain: {self.gain:.4f}"
            )

            if conv < self.epsilon:
                logger.info(
                    f"Convergence threshold reached at iteration {self.iteration}"
                )
                break

            if (
                self.is_checkpointing_enabled
                and self.iteration % self.checkpoint_frequency == 0
            ):
                self.save(self.iteration)

        if conv >= self.epsilon:
            logger.info("Maximum iterations reached")

        # Final checkpoint if enabled
        if self.is_checkpointing_enabled:
            self.save(self.iteration)

        # Extract policy if converged or on final iteration
        logger.info("Extracting policy")
        self.policy = self._extract_policy()
        logger.info("Policy extracted")

        logger.success("Relative value iteration completed")
        return self.solver_state

    def _get_solver_config(self) -> RelativeValueIterationConfig:
        """Get solver configuration for reconstruction.

        Returns:
            Configuration containing all parameters needed to reconstruct
            this solver instance
        """
        return RelativeValueIterationConfig(
            problem=self.problem.config,
            epsilon=self.epsilon,
            max_batch_size=self.max_batch_size,
            jax_double_precision=self.jax_double_precision,
            checkpoint_dir=str(self.checkpoint_dir) if self.checkpoint_dir else None,
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

    def _restore_state_from_checkpoint(
        self, solver_state: RelativeValueIterationState
    ) -> None:
        """Restore solver state from checkpoint."""
        self.values = solver_state.values
        self.policy = solver_state.policy
        self.iteration = solver_state.info.iteration
        self.gain = solver_state.info.gain
