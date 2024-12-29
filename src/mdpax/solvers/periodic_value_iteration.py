"""Value iteration solver with periodic convergence checking."""

from pathlib import Path
from typing import Optional, Tuple, Union

import chex
import jax.numpy as jnp
import numpy as np
from hydra.conf import MISSING, dataclass
from loguru import logger

from mdpax.core.problem import Problem
from mdpax.core.solver import SolverInfo, SolverState
from mdpax.solvers.value_iteration import ValueIteration, ValueIterationConfig


@dataclass
class PeriodicValueIterationConfig(ValueIterationConfig):
    """Configuration for the Periodic Value Iteration solver."""

    _target_: str = "mdpax.solvers.periodic_value_iteration.PeriodicValueIteration"
    period: int = MISSING  # Period for convergence checking
    clear_value_history_on_convergence: bool = True


@chex.dataclass(frozen=True)
class PeriodicValueIterationInfo(SolverInfo):
    """Runtime information for periodic value iteration."""

    value_history: chex.Array  # [period + 1, n_states]
    history_index: int
    period: int


@chex.dataclass(frozen=True)
class PeriodicValueIterationState(SolverState):
    """Runtime state for periodic value iteration."""

    info: PeriodicValueIterationInfo


class PeriodicValueIteration(ValueIteration):
    """Value iteration solver that checks for convergence over a specified period.

    This solver extends standard value iteration to handle problems where the
    value function may oscillate with a known period rather than converging
    to a single value. This is particularly useful for:

    1. Cyclic MDPs where optimal values naturally oscillate
    2. Problems with periodic structure in state/action space
    3. Cases where standard VI fails to converge

    For discounted problems (γ < 1), it accounts for discounting when comparing
    values across periods. For undiscounted problems (γ = 1), it directly
    compares values separated by one period.
    """

    def __init__(
        self,
        problem: Problem,
        period: int,
        gamma: float = 0.99,
        max_iter: int = 1000,
        epsilon: float = 1e-3,
        max_batch_size: int = 1024,
        jax_double_precision: bool = True,
        verbose: int = 2,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_frequency: int = 0,
        max_checkpoints: int = 1,
        enable_async_checkpointing: bool = True,
        clear_value_history_on_convergence: bool = True,
    ):
        """Initialize solver.

        Args:
            problem: MDP problem to solve
            period: Expected period length for value function oscillation
            gamma: Discount factor (use 1.0 for average reward case)
            max_iter: Maximum number of iterations
            epsilon: Convergence threshold
            max_batch_size: Size of state batches for parallel processing
            verbose: Verbosity level
            checkpoint_dir: Directory for checkpoints (optional)
            checkpoint_frequency: How often to save checkpoints (iterations)
            max_checkpoints: Maximum checkpoints to keep
            enable_async_checkpointing: Whether to checkpoint asynchronously
            clear_value_history_on_convergence: Whether to clear value history
                after convergence
        """
        # Validate inputs
        if period <= 0:
            raise ValueError("Period must be positive")
        if gamma == 1.0 and period < 2:
            raise ValueError("Period must be at least 2 for undiscounted case")

        self.period = period
        self.clear_value_history_on_convergence = clear_value_history_on_convergence

        super().__init__(
            problem,
            gamma=gamma,
            max_iter=max_iter,
            epsilon=epsilon,
            max_batch_size=max_batch_size,
            jax_double_precision=jax_double_precision,
            verbose=verbose,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=checkpoint_frequency,
            max_checkpoints=max_checkpoints,
            enable_async_checkpointing=enable_async_checkpointing,
        )

        # Initialize value history in CPU memory
        self.value_history = np.zeros((period + 1, problem.n_states))
        self.history_index: int = 0
        self.value_history[0] = np.array(self.values)

    def _iteration_step(self) -> Tuple[jnp.ndarray, float]:
        """Run one iteration and check for periodic convergence.

        The convergence measure is the span (max - min) of the period deltas.
        For undiscounted problems, this is simply the span of differences between
        current values and values from one period ago. For discounted problems,
        we sum the consecutive differences over the period, scaling each by the
        appropriate discount factor to get undiscounted changes.

        Returns:
            Tuple of (new_values, convergence_measure)
        """
        # Get new values using parent's batch processing
        new_values = self._update_values(
            self.batched_states,
            self.problem.action_space,
            self.problem.random_event_space,
            self.gamma,
            self.values,
        )
        # Store values in history (CPU)
        self.history_index = (self.history_index + 1) % (self.period + 1)
        self.value_history[self.history_index] = np.array(new_values)

        # Check periodic convergence if we have enough history
        if self.iteration >= self.period:
            if self.gamma == 1.0:
                min_delta, max_delta = self._calculate_period_deltas_without_discount(
                    new_values
                )
            else:
                min_delta, max_delta = self._calculate_period_deltas_with_discount(
                    new_values, self.gamma
                )
            # Convergence measure is the span of period deltas
            conv = max_delta - min_delta
        else:
            # Not enough history yet
            conv = float("inf")

        return new_values, conv

    def _get_solver_config(self) -> PeriodicValueIterationConfig:
        """Get solver configuration for reconstruction."""
        return PeriodicValueIterationConfig(
            problem=self.problem.get_problem_config(),
            period=self.period,
            gamma=float(self.gamma),
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            max_batch_size=self.max_batch_size,
            jax_double_precision=self.jax_double_precision,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_frequency=self.checkpoint_frequency,
            max_checkpoints=self.max_checkpoints,
            enable_async_checkpointing=self.enable_async_checkpointing,
            clear_value_history_on_convergence=self.clear_value_history_on_convergence,
        )

    @property
    def solver_state(self) -> PeriodicValueIterationState:
        """Get solver state for checkpointing."""
        return PeriodicValueIterationState(
            values=self.values,
            policy=self.policy,
            info=PeriodicValueIterationInfo(
                iteration=self.iteration,
                value_history=self.value_history,
                history_index=self.history_index,
                period=self.period,
            ),
        )

    def _restore_from_checkpoint(
        self, solver_state: PeriodicValueIterationState
    ) -> None:
        """Restore solver state from checkpoint."""
        self.values = solver_state.values
        self.policy = solver_state.policy
        self.iteration = solver_state.info.iteration
        self.value_history = solver_state.info.value_history
        self.history_index = solver_state.info.history_index
        self.period = solver_state.info.period

    def _clear_value_history(self) -> None:
        """Clear value history to free memory after convergence."""
        if self.clear_value_history_on_convergence:
            self.value_history = None

    def solve(self) -> PeriodicValueIterationState:
        """Solve the MDP and optionally clear history."""
        while self.iteration < self.max_iter:
            self.iteration += 1

            # Perform iteration step
            logger.debug(f"Value shape: {self.values.shape}")
            new_values, conv = self._iteration_step()

            # Update values and iteration count
            self.values = new_values

            logger.info(f"Iteration {self.iteration}: delta_diff: {conv:.4f}")

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
        self.policy = self._extract_policy()
        logger.info("Policy extracted")

        logger.info("Periodic value iteration completed")
        self._clear_value_history()
        return self.solver_state

    def _calculate_period_deltas_without_discount(
        self, values: jnp.ndarray
    ) -> Tuple[float, float]:
        """Return min and max value changes over a period without discounting.

        For problems without discounting (γ=1), we simply compare current values
        with values from one period ago. The circular buffer is arranged so that
        index (i+1) % (period+1) contains values from one period before index i.

        Args:
            values: Current value function

        Returns:
            Tuple of (min_delta, max_delta) over the period
        """
        # Get values from one period ago
        prev_index = (self.history_index + 1) % (self.period + 1)
        values_prev = jnp.array(self.value_history[prev_index])

        # Compute value differences
        value_diffs = values - values_prev
        return jnp.min(value_diffs), jnp.max(value_diffs)

    def _calculate_period_deltas_with_discount(
        self, values: jnp.ndarray, gamma: float
    ) -> Tuple[float, float]:
        """Return min and max undiscounted value changes over a period with discounting.

        For discounted problems (γ<1), we need to sum the differences between
        consecutive steps in the period, adjusting for the discount factor.
        The differences are scaled by 1/γ^(current_iteration - p - 1) where p is
        the position in the period, effectively removing the discount factor's
        effect on the period delta.

        Args:
            values: Current value function
            gamma: Discount factor

        Returns:
            Tuple of (min_delta, max_delta) over the period
        """
        # Initialize on CPU to avoid GPU memory overhead
        period_deltas = np.zeros_like(values)

        # Process one step at a time to minimize memory usage
        for p in range(self.period):
            curr_index = (self.history_index - p) % (self.period + 1)
            prev_index = (curr_index - 1) % (self.period + 1)

            # Get values and compute difference
            values_curr = self.value_history[curr_index]
            values_prev = self.value_history[prev_index]

            # Update period deltas on CPU
            period_deltas += (values_curr - values_prev) / (
                gamma ** (self.iteration - p - 1)
            )

        # Final reduction on GPU
        period_deltas = jnp.array(period_deltas)
        return jnp.min(period_deltas), jnp.max(period_deltas)
