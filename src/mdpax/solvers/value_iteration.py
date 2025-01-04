"""Value iteration solver for MDPs."""

import jax
import jax.numpy as jnp
from hydra.conf import dataclass
from hydra.utils import instantiate
from jaxtyping import Array, Float
from loguru import logger

from mdpax.core.problem import Problem, ProblemConfig
from mdpax.core.solver import (
    Solver,
    SolverConfig,
    SolverState,
)
from mdpax.utils.checkpointing import CheckpointMixin
from mdpax.utils.logging import get_convergence_format
from mdpax.utils.types import (
    ActionSpace,
    ActionVector,
    BatchedStates,
    RandomEventSpace,
    StateBatch,
    StateVector,
)


@dataclass
class ValueIterationConfig(SolverConfig):
    """Configuration for the Value Iteration solver.

    This solver performs synchronous updates over all states using
    parallel processing across devices.

    Args:
        problem: Optional problem configuration. If not provided, can pass a Problem
            instance directly to the solver. If a Problem instance with a config is
            provided to the solver, its config will be extracted and stored here.
            Must be a ProblemConfig if provided.
        gamma: Discount factor in [0,1]
        epsilon: Convergence threshold for value changes (must be positive)
        max_batch_size: Maximum states to process in parallel on each device (must be positive)
        jax_double_precision: Whether to use float64 precision
        verbose: Logging verbosity level (must be 0-4)
        checkpoint_dir: Directory to store checkpoints
        checkpoint_frequency: How often to save checkpoints (must be non-negative, 0 to disable)
        max_checkpoints: Maximum number of checkpoints to keep (must be non-negative)
        enable_async_checkpointing: Whether to save checkpoints asynchronously

    Example:
        >>> # Using a Problem instance with config
        >>> problem = Forest(S=4)  # Has config
        >>> solver = ValueIteration(problem=problem)  # Config extracted automatically

        >>> # Or using a ProblemConfig directly
        >>> problem_config = ForestConfig(S=4)
        >>> config = ValueIterationConfig(problem=problem_config)
        >>> solver = ValueIteration(config=config)

        >>> # Or using a Problem instance without config
        >>> problem = CustomProblem()  # No config
        >>> solver = ValueIteration(problem=problem)  # Checkpointing will be disabled
    """

    _target_: str = "mdpax.solvers.value_iteration.ValueIteration"
    problem: ProblemConfig | None = None
    gamma: float = 0.99
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
        if self.problem is not None and not isinstance(self.problem, ProblemConfig):
            raise TypeError("problem must be a ProblemConfig if provided")
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


class ValueIteration(Solver, CheckpointMixin):
    """Value iteration solver for MDPs.

    This solver implements synchronous value iteration with parallel state updates
    across devices. States are automatically batched and padded for efficient
    parallel processing.

    Convergence testing follows pymdptoolbox, testing the span of differences
    in values instead of the maximum absolute difference. The convergence
    threshold is adjusted for the discount factor gamma.

    Supports checkpointing for long-running problems using the CheckpointMixin.

    Args:
        problem: Problem instance or None if using config
        config: Configuration object. If provided, other kwargs are ignored.
        **kwargs: Parameters matching :class:`ValueIterationConfig`.
            See Config class for detailed parameter descriptions.
    """

    Config = ValueIterationConfig

    def __init__(
        self,
        problem: Problem | None = None,
        config: ValueIterationConfig | None = None,
        **kwargs,
    ):
        """Initialize the solver."""
        if config is not None:
            self.config = config
        else:
            self.config = self.Config(**kwargs)

        # Handle problem instance vs config
        if problem is not None:
            # If given a Problem instance directly, store
            # config, if it has one
            if hasattr(problem, "config"):
                self.config.problem = problem.config
        else:
            # If no problem instance, must have config
            if self.config.problem is None:
                raise ValueError("Must provide either problem instance or config")
            problem = instantiate(self.config.problem)

        super().__init__(
            problem,
            self.config.gamma,
            self.config.epsilon,
            self.config.max_batch_size,
            self.config.jax_double_precision,
            self.config.verbose,
        )
        self._setup_checkpointing(
            self.config.checkpoint_dir,
            self.config.checkpoint_frequency,
            max_checkpoints=self.config.max_checkpoints,
            enable_async_checkpointing=self.config.enable_async_checkpointing,
        )
        self.policy = None

    def _setup_solver(self) -> None:
        """Setup solver-specific data structures and functions.

        Sets up:
            - Convergence threshold based on gamma and epsilon
            - Convergence logging format
            - Pmapped functions for parallel value updates and policy extraction

        The convergence threshold is computed as:
            - epsilon if gamma == 1
            - epsilon * (1 - gamma) / gamma otherwise
        following mdptoolbox's implementation.

        Returns:
            None
        """

        self._thresh = (
            self.epsilon
            if self.gamma == 1
            else self.epsilon * (1 - self.gamma) / self.gamma
        )

        # Get convergence format for logging convergence metrics
        self.convergence_format = get_convergence_format(float(self._thresh))

        self._calculate_updated_value_scan_state_batches_pmap = jax.pmap(
            self._calculate_updated_value_scan_state_batches,
            in_axes=((None, None, None, None), 0),
        )

        self._extract_policy_idx_scan_state_batches_pmap = jax.pmap(
            self._extract_policy_idx_scan_state_batches,
            in_axes=((None, None, None, None), 0),
        )

    def _get_value_next_state(
        self, next_state: StateVector, values: Float[Array, "n_states"]
    ) -> float:
        """Lookup the value of the next state in the value function.

        Args:
            next_state: State vector to look up [state_dim]
            values: Current value function [n_states]

        Returns:
            Value of the next state
        """
        return values[self.problem.state_to_index(next_state)]

    def _calculate_updated_state_action_value(
        self,
        state: StateVector,
        action: ActionVector,
        random_events: RandomEventSpace,
        gamma: float,
        values: Float[Array, "n_states"],
    ) -> float:
        """Calculate the expected value for a state-action pair.

        Args:
            state: Current state vector [state_dim]
            action: Action vector [action_dim]
            random_events: All possible random events [n_events, event_dim]
            gamma: Discount factor
            values: Current value function [n_states]

        Returns:
            Expected value for the state-action pair
        """
        next_states, single_step_rewards = jax.vmap(
            self.problem.transition,
            in_axes=(None, None, 0),
        )(
            state,
            action,
            random_events,
        )
        next_state_values = jax.vmap(
            self._get_value_next_state,
            in_axes=(0, None),
        )(next_states, values)
        probs = jax.vmap(
            self.problem.random_event_probability,
            in_axes=(None, None, 0),
        )(state, action, random_events)
        return (single_step_rewards + gamma * next_state_values).dot(probs)

    def _calculate_updated_value(
        self,
        state: StateVector,
        actions: ActionSpace,
        random_events: RandomEventSpace,
        gamma: float,
        values: Float[Array, "n_states"],
    ) -> float:
        """Calculate the maximum expected value over all actions for a state.

        Args:
            state: Current state vector [state_dim]
            actions: All possible actions [n_actions, action_dim]
            random_events: All possible random events [n_events, event_dim]
            gamma: Discount factor
            values: Current value function [n_states]

        Returns:
            Maximum expected value over all actions
        """
        return jnp.max(
            jax.vmap(
                self._calculate_updated_state_action_value,
                in_axes=(None, 0, None, None, None),
            )(state, actions, random_events, gamma, values)
        )

    def _calculate_updated_value_state_batch(
        self,
        carry: tuple[Float[Array, "n_states"], float, ActionSpace, RandomEventSpace],
        state_batch: StateBatch,
    ) -> tuple[tuple, Float[Array, "batch_size"]]:
        """Calculate updated values for a batch of states.

        Args:
            carry: Tuple of (values, gamma, action_space, random_event_space)
            state_batch: Batch of states to update [batch_size, state_dim]

        Returns:
            Tuple of (carry, new_values) where new_values has shape [batch_size]
        """
        values, gamma, action_space, random_event_space = carry
        new_values = jax.vmap(
            self._calculate_updated_value,
            in_axes=(0, None, None, None, None),
        )(state_batch, values, gamma, action_space, random_event_space)
        return carry, new_values

    def _calculate_updated_value_scan_state_batches(
        self,
        carry: tuple[Float[Array, "n_states"], float, ActionSpace, RandomEventSpace],
        padded_batched_states: BatchedStates,
    ) -> Float[Array, "n_devices n_batches batch_size"]:
        """Update values for multiple batches of states.

        Uses jax.lax.scan to loop over batches efficiently.

        Args:
            carry: Tuple of (actions, random_events, gamma, values)
            padded_batched_states: States prepared for batch processing
                Shape: [n_devices, n_batches, batch_size, state_dim]

        Returns:
            Array of updated values for all states [n_devices, n_batches, batch_size]
        """
        _, new_values_padded = jax.lax.scan(
            self._calculate_updated_value_state_batch,
            carry,
            padded_batched_states,
        )
        return new_values_padded

    def _extract_policy_idx_one_state(
        self,
        state: StateVector,
        actions: ActionSpace,
        random_events: RandomEventSpace,
        gamma: float,
        values: Float[Array, "n_states"],
    ) -> int:
        """Find the optimal action index for a single state.

        Args:
            state: Current state vector [state_dim]
            actions: All possible actions [n_actions, action_dim]
            random_events: All possible random events [n_events, event_dim]
            gamma: Discount factor
            values: Current value function [n_states]

        Returns:
            Index of the optimal action
        """
        best_action_idx = jnp.argmax(
            jax.vmap(
                self._calculate_updated_state_action_value,
                in_axes=(None, 0, None, None, None),
            )(state, actions, random_events, gamma, values)
        )
        return best_action_idx

    def _extract_policy_idx_state_batch(
        self,
        carry: tuple[ActionSpace, RandomEventSpace, float, Float[Array, "n_states"]],
        state_batch: StateBatch,
    ) -> tuple[tuple, Float[Array, "batch_size"]]:
        """Extract optimal action indices for a batch of states.

        Args:
            carry: Tuple of (actions, random_events, gamma, values)
            state_batch: Batch of states [batch_size, state_dim]

        Returns:
            Tuple of (carry, action_indices) where action_indices has shape [batch_size]
        """
        actions, random_events, gamma, values = carry
        best_action_idxs = jax.vmap(
            self._extract_policy_idx_one_state,
            in_axes=(0, None, None, None, None),
        )(state_batch, actions, random_events, gamma, values)
        return carry, best_action_idxs

    def _extract_policy_idx_scan_state_batches(
        self,
        carry: tuple[Float[Array, "n_states"], float, ActionSpace, RandomEventSpace],
        padded_batched_states: BatchedStates,
    ) -> Float[Array, "n_devices n_batches batch_size"]:
        """Extract optimal action indices for multiple batches of states.

        Uses jax.lax.scan to loop over batches efficiently.

        Args:
            carry: Tuple of (actions, random_events, gamma, values)
            padded_batched_states: States prepared for batch processing
                Shape: [n_devices, n_batches, batch_size, state_dim]

        Returns:
            Array of updated values for all states [n_devices, n_batches, batch_size]
        """
        _, best_action_idxs_padded = jax.lax.scan(
            self._extract_policy_idx_state_batch,
            carry,
            padded_batched_states,
        )
        return best_action_idxs_padded

    def _iteration_step(self) -> tuple[Float[Array, "n_states"], float]:
        """Perform one iteration of the solution algorithm.

        Returns:
            Tuple of (new values, convergence measure) where new values has shape
            [n_states]
        """
        new_values = self._update_values(
            self.batched_states,
            self.problem.action_space,
            self.problem.random_event_space,
            self.gamma,
            self.values,
        )

        # Calculate convergence measure, span of differences in values
        # as in mdptoolbox
        # https://github.com/sawcordwell/pymdptoolbox/blob/master/src/mdptoolbox/mdp.py
        # https://github.com/sawcordwell/pymdptoolbox/blob/master/src/mdptoolbox/util.py
        conv = self._get_span(new_values, self.values)
        return new_values, conv

    def _get_span(
        self, new_values: Float[Array, "n_states"], old_values: Float[Array, "n_states"]
    ) -> float:
        """Get the span of differences in values.

        The span is defined as max(delta) - min(delta) where delta is the
        difference between new and old values. This is used as the convergence
        measure following pymdptoolbox's implementation.

        Args:
            new_values: Updated value function [n_states]
            old_values: Previous value function [n_states]

        Returns:
            Span (max - min) of value differences
        """
        delta = new_values - old_values
        return jnp.max(delta) - jnp.min(delta)

    def _update_values(
        self,
        batched_states: BatchedStates,
        actions: ActionSpace,
        random_events: RandomEventSpace,
        gamma: float,
        values: Float[Array, "n_states"],
    ) -> Float[Array, "n_states"]:
        """Update values for a batch of states using parallel processing.

        Computes new values by:
        1. Calculating updated values for each state-action-event combination
        2. Processing states in parallel across devices using pmap
        3. Unbatching and removing padding from results

        Args:
            batched_states: Batched state vectors [n_devices, n_batches, batch_size, state_dim]
            actions: Action space [n_actions, action_dim]
            random_events: Random event space [n_events, event_dim]
            gamma: Discount factor
            values: Current value function [n_states]

        Returns:
            Array of updated values [n_states]
        """
        padded_batched_values = self._calculate_updated_value_scan_state_batches_pmap(
            (actions, random_events, gamma, values), batched_states
        )
        new_values = self._unbatch_results(padded_batched_values)
        return new_values

    def _extract_policy(
        self,
    ) -> Float[Array, "n_states action_dim"]:
        """Extract the optimal policy from the current value function.

        Returns:
            Array of optimal actions for each state [n_states, action_dim]
        """
        padded_batched_policy_idxs = self._extract_policy_idx_scan_state_batches_pmap(
            (
                self.problem.action_space,
                self.problem.random_event_space,
                self.gamma,
                self.values,
            ),
            self.batched_states,
        )
        policy_idxs = self._unbatch_results(padded_batched_policy_idxs)
        return jnp.take(self.problem.action_space, policy_idxs, axis=0)

    def solve(self, max_iterations: int = 2000) -> SolverState:
        """Run solver to convergence or max iterations.

        Args:
            max_iterations: Maximum number of iterations to run

        Returns:
            SolverState containing final values [n_states], optimal policy [n_states, action_dim],
            and SolverInfo including iteration count
        """
        for _ in range(max_iterations):
            self.iteration += 1
            new_values, conv = self._iteration_step()
            self.values = new_values

            logger.info(
                f"Iteration {self.iteration} span: {conv:{self.convergence_format}}"
            )

            if conv < self._thresh:
                logger.info(
                    f"Convergence threshold reached at iteration {self.iteration}"
                )
                break

            if (
                self.is_checkpointing_enabled
                and self.iteration % self.checkpoint_frequency == 0
            ):
                self.save(self.iteration)

        if conv >= self._thresh:
            logger.info("Maximum iterations reached")

        # Final checkpoint if enabled
        if self.is_checkpointing_enabled:
            self.save(self.iteration)

        # Extract policy if converged or on final iteration
        logger.info("Extracting policy")
        self.policy = self._extract_policy()
        logger.info("Policy extracted")

        logger.success("Value iteration completed")
        return self.solver_state

    def _restore_state_from_checkpoint(self, solver_state: SolverState) -> None:
        """Restore solver state from checkpoint."""
        self.values = solver_state.values
        self.policy = solver_state.policy
        self.iteration = solver_state.info.iteration
