"""Relative value iteration solver for average reward MDPs."""

import chex
from hydra.conf import dataclass
from loguru import logger

from mdpax.core.problem import Problem, ProblemConfig
from mdpax.core.solver import SolverConfig, SolverInfo, SolverState
from mdpax.solvers.value_iteration import ValueIteration
from mdpax.utils.logging import get_convergence_format
from mdpax.utils.types import ValueFunction


@dataclass
class RelativeValueIterationConfig(SolverConfig):
    """Configuration for the Relative Value Iteration solver.

    Args:
        problem: Optional problem configuration. If not provided, can pass a Problem
            instance directly to the solver. If a Problem instance with a config is
            provided to the solver, its config will be extracted and stored here.
        epsilon: Convergence threshold for value changes
        gamma: Discount factor (fixed at 1.0 for relative value iteration)
        max_batch_size: Maximum states to process in parallel on each device
        jax_double_precision: Whether to use float64 precision
        verbose: Logging verbosity level (0-4)
        checkpoint_dir: Directory to store checkpoints
        checkpoint_frequency: How often to save checkpoints (0 to disable)
        max_checkpoints: Maximum number of checkpoints to keep
        enable_async_checkpointing: Whether to save checkpoints asynchronously

    Example:
        >>> # Using a Problem instance with config
        >>> problem = Forest(S=4)  # Has config
        >>> solver = RelativeValueIteration(problem=problem)  # Config extracted automatically

        >>> # Or using a ProblemConfig directly
        >>> problem_config = ForestConfig(S=4)
        >>> config = RelativeValueIterationConfig(problem=problem_config)
        >>> solver = RelativeValueIteration(config=config)

        >>> # Or using a Problem instance without config
        >>> problem = CustomProblem()  # No config
        >>> solver = RelativeValueIteration(problem=problem)  # Checkpointing will be disabled
    """

    _target_: str = "mdpax.solvers.relative_value_iteration.RelativeValueIteration"
    problem: ProblemConfig | None = None
    gamma: float = 1.0
    epsilon: float = 1e-3
    max_batch_size: int = 1024
    jax_double_precision: bool = True
    verbose: int = 2
    checkpoint_dir: str | None = None
    checkpoint_frequency: int = 0
    max_checkpoints: int = 1
    enable_async_checkpointing: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.problem is not None and not isinstance(ProblemConfig):
            raise TypeError("problem must be a ProblemConfig if provided")
        if not self.gamma == 1.0:
            raise ValueError("gamma must be 1.0 for relative value iteration")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.checkpoint_frequency < 0:
            raise ValueError("checkpoint_frequency must be non-negative")
        if self.max_checkpoints < 0:
            raise ValueError("max_checkpoints must be non-negative")


@chex.dataclass(frozen=True)
class RelativeValueIterationInfo(SolverInfo):
    """Runtime information for relative value iteration.

    Attributes:
        gain: Current gain term for value function adjustment,
            converges to average reward per timestep
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
        problem: Problem instance or None if using config
        config: Configuration object. If provided, other kwargs are ignored.
        **kwargs: Parameters matching :class:`RelativeValueIterationConfig`.
            See Config class for detailed parameter descriptions.
    """

    Config = RelativeValueIterationConfig

    def __init__(
        self,
        problem: Problem | None = None,
        config: RelativeValueIterationConfig | None = None,
        **kwargs,
    ):
        """Initialize solver."""

        super().__init__(problem=problem, config=config, **kwargs)
        # Get convergence format for logging convergence metrics
        self.convergence_format = get_convergence_format(self.config.epsilon)
        self.gain = 0.0

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
        new_values = new_values - self.gain

        span = self._get_span(new_values, self.values)

        self.gain = new_values[-1]

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
        return self.config

    @property
    def solver_state(self) -> RelativeValueIterationState:
        """Get solver state for checkpointing."""
        return RelativeValueIterationState(
            values=self.values,
            policy=self.policy,
            info=RelativeValueIterationInfo(
                iteration=self.iteration,
                gain=self.gain,
            ),
        )

    def _restore_state_from_checkpoint(
        self, solver_state: RelativeValueIterationState
    ) -> None:
        """Restore solver state from checkpoint."""
        self.values = solver_state.values
        self.policy = solver_state.policy
        self.iteration = solver_state.info.iteration
        self.gain = solver_state.info.gain
