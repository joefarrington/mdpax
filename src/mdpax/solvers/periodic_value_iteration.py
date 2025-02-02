"""Value iteration solver with periodic convergence checking."""

import chex
import jax.numpy as jnp
import numpy as np
from hydra.conf import MISSING, dataclass
from jaxtyping import Array, Float
from loguru import logger

from mdpax.core.problem import Problem, ProblemConfig
from mdpax.core.solver import SolverConfig, SolverInfo, SolverState
from mdpax.solvers.value_iteration import ValueIteration
from mdpax.utils.logging import get_convergence_format
from mdpax.utils.types import (
    ValueFunction,
)


@dataclass
class PeriodicValueIterationConfig(SolverConfig):
    """Configuration for the Periodic Value Iteration solver.

    This solver extends value iteration to check for convergence over a
    specified period length rather than between consecutive iterations.

    Args:
        problem: Optional problem configuration. If not provided, can pass a Problem
            instance directly to the solver. If a Problem instance with a config is
            provided to the solver, its config will be extracted and stored here.
        period: Number of iterations to check for periodic convergence (must be positive)
        gamma: Discount factor in [0,1]
        epsilon: Convergence threshold for value changes (must be positive)
        max_batch_size: Maximum states to process in parallel on each device (must be positive)
        jax_double_precision: Whether to use float64 precision
        verbose: Logging verbosity level (must be 0-4)
        checkpoint_dir: Directory to store checkpoints
        checkpoint_frequency: How often to save checkpoints (must be non-negative, 0 to disable)
        max_checkpoints: Maximum number of checkpoints to keep (must be non-negative)
        enable_async_checkpointing: Whether to save checkpoints asynchronously
        clear_value_history_on_convergence: Whether to clear history after convergence

    Example:
        >>> # Using a Problem instance with config
        >>> problem = Forest(S=4)  # Has config
        >>> solver = PeriodicValueIteration(problem=problem, period=2)  # Config extracted automatically

        >>> # Or using a ProblemConfig directly
        >>> problem_config = ForestConfig(S=4)
        >>> config = PeriodicValueIterationConfig(problem=problem_config, period=2)
        >>> solver = PeriodicValueIteration(config=config)

        >>> # Or using a Problem instance without config
        >>> problem = CustomProblem()  # No config
        >>> solver = PeriodicValueIteration(problem=problem, period=2)  # Checkpointing will be disabled
    """

    _target_: str = "mdpax.solvers.periodic_value_iteration.PeriodicValueIteration"
    problem: ProblemConfig | None = None
    period: int = MISSING
    gamma: float = 0.99
    epsilon: float = 1e-3
    max_batch_size: int = 1024
    jax_double_precision: bool = True
    verbose: int = 2
    checkpoint_dir: str | None = None
    checkpoint_frequency: int = 0
    max_checkpoints: int = 1
    enable_async_checkpointing: bool = True
    clear_value_history_on_convergence: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.problem is not None and not isinstance(self.problem, ProblemConfig):
            raise TypeError("problem must be a ProblemConfig if provided")
        if self.period <= 0:
            raise ValueError("Period must be positive")
        if self.gamma == 1.0 and self.period < 2:
            raise ValueError("Period must be at least 2 for undiscounted case")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.checkpoint_frequency < 0:
            raise ValueError("checkpoint_frequency must be non-negative")
        if self.max_checkpoints < 0:
            raise ValueError("max_checkpoints must be non-negative")
        if not 0 <= self.verbose <= 4:
            raise ValueError("verbose must be between 0 and 4")


@chex.dataclass(frozen=True)
class PeriodicValueIterationInfo(SolverInfo):
    """Runtime information for periodic value iteration.

    Attributes:
        value_history: History of value functions [period, n_states]
        history_index: Current position in circular history buffer
        period: Length of period being checked
    """

    value_history: Float[Array, "period n_states"]
    history_index: int
    period: int


@chex.dataclass(frozen=True)
class PeriodicValueIterationState(SolverState):
    """Runtime state for periodic value iteration.

    Attributes:
        values: Current value function [n_states]
        policy: Current policy [n_states, action_dim]
        info: Solver metadata including value history
    """

    info: PeriodicValueIterationInfo


class PeriodicValueIteration(ValueIteration):
    """Periodic value iteration solver for MDPs

    This is particularly useful for problems with periodic
    structure in the state space, where it may require fewer iterations
    to reach convergence than standard value iteration.

    Convergence testing is based on the span of value differences over a period.
    For undiscounted problems, this is simply the span of differences between
    current values and values from one period ago. For discounted problems,
    we sum the consecutive differences over the period, scaling each by the
    appropriate discount factor. The solver stores a history of values over
    the period to perform this comparison.

    Supports checkpointing for long-running problems using the CheckpointMixin.

    Args:
        problem: Problem instance or None if using config
        config: Configuration object. If provided, other kwargs are ignored.
        **kwargs: Parameters matching :class:`PeriodicValueIterationConfig`.
            See Config class for detailed parameter descriptions.
    """

    Config = PeriodicValueIterationConfig

    def __init__(
        self,
        problem: Problem | None = None,
        config: PeriodicValueIterationConfig | None = None,
        **kwargs,
    ):
        """Initialize solver."""
        super().__init__(problem=problem, config=config, **kwargs)

    def _setup_config(
        self, problem: Problem, config: SolverConfig | None = None, **kwargs
    ) -> None:
        super()._setup_config(problem, config, **kwargs)
        self.clear_value_history_on_convergence = (
            self.config.clear_value_history_on_convergence
        )
        self.period = self.config.period

    def _setup_convergence_testing(self) -> None:
        """Setup convergence test.

        For periodic value iteration, we use the span of differences over a period
        as the convergence measure. The convergence threshold is simply epsilon
        (no adjustment for gamma needed since we're comparing over a period).
        """
        self._convergence_test_fn = self._get_periodic_span
        self.conv_threshold = self.epsilon
        self._convergence_desc = "period_span"
        # Get convergence format for logging convergence metrics
        self.convergence_format = get_convergence_format(float(self.conv_threshold))

    def _initialize_solver_state_elements(self) -> None:
        """Initialize solver state elements."""
        super()._initialize_solver_state_elements()
        # Initialize value history in CPU memory
        self.value_history = np.zeros((self.period + 1, self.problem.n_states))
        self.history_index: int = 0
        self.value_history[0] = np.array(self.values)

    def _get_periodic_span(
        self,
        new_values: ValueFunction,
        old_values: ValueFunction,
        history_index: int,
        period: int,
        value_history: Float[Array, "period_plus_one n_states"],
        iteration: int,
        gamma: float,
    ) -> float:
        """Calculate convergence measure based on span of differences over a period.

        For undiscounted problems (gamma=1), this is the span of differences between
        current values and values from one period ago. For discounted problems,
        we sum consecutive differences over the period, adjusting for the discount factor.

        Args:
            new_values: Current value function [n_states]
            old_values: Previous value function [n_states] (unused)
            history_index: Current position in circular history buffer
            period: Length of period to check
            value_history: History of values [period+1, n_states]
            iteration: Current iteration number
            gamma: Discount factor

        Returns:
            Span of differences over the period
        """
        if iteration < period:
            return float("inf")

        if gamma == 1.0:
            return self._calculate_period_span_without_discount(
                new_values, history_index, period, value_history
            )
        else:
            return self._calculate_period_span_with_discount(
                new_values, history_index, period, value_history, iteration, gamma
            )

    def _calculate_period_span_without_discount(
        self,
        values: ValueFunction,
        history_index: int,
        period: int,
        value_history: Float[Array, "period_plus_one n_states"],
    ) -> float:
        """Calculate span of value changes over a period without discounting.

        For problems without discounting (gamma=1), we simply compare current values
        with values from one period ago. The circular buffer is arranged so that
        index (i+1) % (period+1) contains values from one period ago.

        Args:
            values: Current value function [n_states]
            history_index: Current position in circular history buffer
            period: Length of period to check
            value_history: History of values [period+1, n_states]

        Returns:
            Span of differences over the period
        """
        prev_index = (history_index + 1) % (period + 1)
        values_prev = jnp.array(value_history[prev_index])
        return self._get_span(values, values_prev)

    def _calculate_period_span_with_discount(
        self,
        values: ValueFunction,
        history_index: int,
        period: int,
        value_history: Float[Array, "period_plus_one n_states"],
        iteration: int,
        gamma: float,
    ) -> float:
        """Calculate span of undiscounted value changes over a period.

        For discounted problems (gamma<1), we sum the differences between consecutive
        steps in the period, adjusting for the discount factor. The differences
        are scaled by 1/gamma^(current_iteration - p - 1) to remove discounting.

        Args:
            values: Current value function [n_states]
            history_index: Current position in circular history buffer
            period: Length of period to check
            value_history: History of values [period+1, n_states]
            iteration: Current iteration number
            gamma: Discount factor

        Returns:
            Span of differences over the period
        """
        period_deltas = np.zeros_like(values)

        for p in range(period):
            curr_index = (history_index - p) % (period + 1)
            prev_index = (curr_index - 1) % (period + 1)

            values_curr = value_history[curr_index]
            values_prev = value_history[prev_index]

            period_deltas += (values_curr - values_prev) / (
                gamma ** (iteration - p - 1)
            )

        period_deltas = jnp.array(period_deltas)
        return jnp.max(period_deltas) - jnp.min(period_deltas)

    def _iteration_step(self) -> tuple[ValueFunction, float]:
        """Perform one iteration of the solution algorithm.

        Returns:
            Tuple of (new values, convergence measure) where new values has shape
            [n_states]
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

        # Calculate convergence using the test function
        conv = self._convergence_test_fn(
            new_values,
            self.values,
            self.history_index,
            self.period,
            self.value_history,
            self.iteration,
            self.gamma,
        )

        return new_values, conv

    def solve(self, max_iterations: int = 2000) -> PeriodicValueIterationState:
        """Run solver to convergence or max iterations.

        Args:
            max_iterations: Maximum number of iterations to run

        Returns:
            SolverState containing final values [n_states], optimal policy [n_states, action_dim],
            and SolverInfo including iteration count and value history.
        """
        for _ in range(max_iterations):
            self.iteration += 1
            new_values, conv = self._iteration_step()
            self.values = new_values

            logger.info(
                f"Iteration {self.iteration}: {self._convergence_desc}: {conv:{self.convergence_format}}"
            )

            # Check convergence
            if conv < self.conv_threshold:
                logger.info(
                    f"Convergence threshold reached at iteration {self.iteration}"
                )
                break

            if (
                self.is_checkpointing_enabled
                and self.iteration % self.checkpoint_frequency == 0
            ):
                self.save(self.iteration)

        if conv >= self.conv_threshold:
            logger.info("Maximum iterations reached")

        # Final checkpoint if enabled
        if self.is_checkpointing_enabled:
            self.save(self.iteration)

        # Extract policy if converged or on final iteration
        logger.info("Extracting policy")
        self.policy = self._extract_policy()
        logger.info("Policy extracted")

        logger.success("Periodic value iteration completed")
        if conv < self.conv_threshold:
            self._clear_value_history()
        return self.solver_state

    def _clear_value_history(self) -> None:
        """Clear value history to free memory after convergence."""
        if self.clear_value_history_on_convergence:
            self.value_history = None

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

    def _restore_state_from_checkpoint(
        self, solver_state: PeriodicValueIterationState
    ) -> None:
        """Restore solver state from checkpoint."""
        self.values = solver_state.values
        self.policy = solver_state.policy
        self.iteration = solver_state.info.iteration
        self.value_history = solver_state.info.value_history
        self.history_index = solver_state.info.history_index
        self.period = solver_state.info.period
