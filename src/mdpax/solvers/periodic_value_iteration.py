"""Value iteration solver with periodic convergence checking."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

from mdpax.core.problem import Problem
from mdpax.solvers.value_iteration import ValueIteration


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
        batch_size: int = 1024,
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
            batch_size: Size of state batches for parallel processing
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

        super().__init__(
            problem,
            gamma=gamma,
            max_iter=max_iter,
            epsilon=epsilon,
            batch_size=batch_size,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=checkpoint_frequency,
            max_checkpoints=max_checkpoints,
            enable_async_checkpointing=enable_async_checkpointing,
        )
        self.period = period
        self.clear_value_history_on_convergence = clear_value_history_on_convergence

        # Initialize value history in CPU memory
        self.value_history = np.zeros((period + 1, problem.n_states))
        self.history_index: int = 0
        self.value_history[0] = np.array(self.values)

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
        new_values = self._process_batches(
            (self.values, self.gamma), self.batched_states
        )
        new_values = self._unpad_results(new_values.reshape(-1))

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

    def _get_checkpoint_state(self) -> Dict:
        """Get solver state for checkpointing.

        Returns:
            Dict containing solver state
        """
        state = super()._get_checkpoint_state()
        state.update(
            {
                "value_history": self.value_history,
                "history_index": self.history_index,
                "period": self.period,
            }
        )
        return state

    def _restore_from_checkpoint(self, state: Dict) -> None:
        """Restore solver state from checkpoint.

        Args:
            state: Dict containing solver state
        """
        super()._restore_from_checkpoint(state)
        self.value_history = state["value_history"]
        self.history_index = state["history_index"]
        self.period = state["period"]

    def _clear_value_history(self) -> None:
        """Clear value history to free memory after convergence."""
        if self.clear_value_history_on_convergence:
            self.value_history = None

    def solve(self, *args, **kwargs):
        """Solve the MDP and optionally clear history."""
        result = super().solve(*args, **kwargs)
        self._clear_value_history()
        return result
