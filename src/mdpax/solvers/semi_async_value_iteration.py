"""Semi-asynchronous value iteration solver with different batch ordering strategies."""

import chex
import jax
import jax.numpy as jnp
import jax.random as random
from hydra.conf import dataclass
from jaxtyping import Array, Float
from loguru import logger

from mdpax.core.problem import Problem, ProblemConfig
from mdpax.core.solver import SolverConfig, SolverInfo, SolverState
from mdpax.solvers.value_iteration import ValueIteration
from mdpax.utils.types import (
    ActionSpace,
    ActionVector,
    BatchedStates,
    RandomEventSpace,
    StateBatch,
    StateVector,
    ValueFunction,
)


@dataclass
class SemiAsyncValueIterationConfig(SolverConfig):
    """Configuration for the Semi-Asynchronous Value Iteration solver.

    This solver performs asynchronous updates over batches of states using
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
        convergence_test: Strategy for testing convergence ("span" or "max_diff")
        shuffle_states: Whether to shuffle state update order each iteration
        random_seed: Random seed for shuffling states
    """

    _target_: str = "mdpax.solvers.semi_async_value_iteration.SemiAsyncValueIteration"
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
    convergence_test: str = "span"
    shuffle_states: bool = False
    random_seed: int = 42

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
        if self.convergence_test not in ["span", "max_diff"]:
            raise ValueError("Convergence test must be 'span' or 'max_diff'")


@chex.dataclass(frozen=True)
class SemiAsyncValueIterationInfo(SolverInfo):
    """Runtime information for semi-async value iteration.

    Attributes:
        batch_order: Current ordering of batches
    """

    batch_order: jnp.ndarray | None


@chex.dataclass(frozen=True)
class SemiAsyncValueIterationState(SolverState):
    """Runtime state for semi-async value iteration.

    Attributes:
        values: Current value function [n_states]
        policy: Current policy [n_states, action_dim]
        info: Solver metadata including batch ordering info
    """

    info: SemiAsyncValueIterationInfo


class SemiAsyncValueIteration(ValueIteration):
    """Semi-asynchronous value iteration solver with flexible batch ordering.

    This solver extends standard value iteration by:
        - Processing states in batches with updated values immediately available
          to subsequent batches on the same device
        - Supporting a fixed or random (shuffle) state update order

    Supports checkpointing for long-running problems using the CheckpointMixin.

    Args:
        problem: Problem instance or None if using config
        config: Configuration object. If provided, other kwargs are ignored.
        **kwargs: Parameters matching :class:`SemiAsyncValueIterationConfig`.
            See Config class for detailed parameter descriptions.
    """

    Config = SemiAsyncValueIterationConfig

    def __init__(
        self,
        problem: Problem | None = None,
        config: SemiAsyncValueIterationConfig | None = None,
        **kwargs,
    ):
        """Initialize the solver."""
        super().__init__(problem=problem, config=config, **kwargs)

    def _setup_config(
        self, problem: Problem, config: SolverConfig | None = None, **kwargs
    ) -> None:
        super()._setup_config(problem, config, **kwargs)
        self.key = random.PRNGKey(self.config.random_seed)

    def _setup_jax_functions(self) -> None:
        super()._setup_jax_functions()

        # JIT compile core computations
        self._jitted_calculate_updated_state_action_value = jax.jit(
            self._calculate_updated_state_action_value, static_argnums=(0,)
        )

        # JIT compile state shuffling functions
        self._jitted_shuffle_states = jax.jit(self._shuffle_states)
        self._jitted_reorder_values = jax.jit(self._reorder_values)

    def _initialize_solver_state_elements(self) -> None:
        super()._initialize_solver_state_elements()
        self.batch_order = None
        self.inverse_order = None

    def _shuffle_states(
        self, key: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Shuffle the states for random processing order.

        Args:
            key: PRNG key for random shuffling

        Returns:
            Tuple of (shuffled_state_idxs, padded_batched_states, padding_mask)
        """
        # Generate permutation of state indices
        shuffled_state_idxs = jax.random.permutation(
            key, jnp.arange(self.problem.n_states)
        )
        # Shuffle states using these indices
        shuffled_states = self.problem.state_space[shuffled_state_idxs]
        # Prepare batched states from shuffled states
        padded_batched_states = self.batch_processor.prepare_batches(shuffled_states)

        # Create padding mask based on problem size and batch shape
        n_total = (
            padded_batched_states.shape[0]
            * padded_batched_states.shape[1]
            * padded_batched_states.shape[2]
        )
        padding_mask = (jnp.arange(n_total) >= self.problem.n_states).reshape(
            padded_batched_states.shape[0],  # n_devices
            padded_batched_states.shape[1],  # n_batches
            padded_batched_states.shape[2],  # batch_size
        )

        return shuffled_state_idxs, padded_batched_states, padding_mask

    def _reorder_values(
        self, shuffled_state_idxs: jnp.ndarray, values: jnp.ndarray
    ) -> jnp.ndarray:
        """Reorder values back to original state order.

        Args:
            shuffled_state_idxs: Indices used to shuffle states
            values: Values in shuffled order

        Returns:
            Values reordered to match original state order
        """
        return values[jnp.argsort(shuffled_state_idxs)]

    def _get_value_next_state(
        self, next_state: StateVector, values: ValueFunction
    ) -> float:
        """Lookup the value of the next state in the value function.

        Args:
            next_state: Next state vector [state_dim]
            values: Current value function [n_states]

        Returns:
            Value of the next state
        """
        # Values are always in their original positions, so just get the index
        return values[self.problem.state_to_index(next_state)]

    def _calculate_updated_state_action_value(
        self,
        state: StateVector,
        action: ActionVector,
        random_events: RandomEventSpace,
        gamma: float,
        values: ValueFunction,
    ) -> float:
        """Calculate the expected value for a state-action pair.

        Similar to value iteration but uses _map_state_indices for next state lookups
        to handle batch-aware indexing.

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
        )(state, action, random_events)

        # Map indices for correct value lookups in batch context
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
        values: ValueFunction,
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
        carry: tuple[ActionSpace, RandomEventSpace, float, ValueFunction],
        state_batch: StateBatch,
    ) -> tuple[tuple, Float[Array, "batch_size"]]:
        """Calculate updated values for a batch of states.

        Similar to value iteration but ensures values reflect most recent updates.

        Args:
            carry: Tuple of (actions, random_events, gamma, values)
            state_batch: Batch of states to update [batch_size, state_dim]

        Returns:
            Tuple of (carry, new_values) where new_values has shape [batch_size]
        """
        actions, random_events, gamma, values = carry
        new_values = jax.vmap(
            self._calculate_updated_value,
            in_axes=(0, None, None, None, None),
        )(state_batch, actions, random_events, gamma, values)
        return carry, new_values

    def _batch_get_indices(self, state_batch: StateBatch) -> jnp.ndarray:
        """Get indices for a batch of states.

        Args:
            state_batch: Batch of states [batch_size, state_dim]

        Returns:
            Array of indices [batch_size]
        """
        return jax.vmap(self.problem.state_to_index)(state_batch)

    def _calculate_updated_value_scan_state_batches(
        self,
        carry: tuple[ActionSpace, RandomEventSpace, float, ValueFunction],
        batched_input: tuple[BatchedStates, jnp.ndarray],
    ) -> Float[Array, "n_devices n_batches batch_size"]:
        """Update values for multiple batches of states.

        Key difference from value iteration: values in carry are updated after each batch
        to make them available for subsequent batches. The batch order determines the
        sequence of processing, but values are always stored in their original positions
        in the state space.

        Args:
            carry: Tuple of (actions, random_events, gamma, values)
            batched_input: Tuple of (batched_states, padding_mask) where:
                - batched_states has shape [n_devices, n_batches, batch_size, state_dim]
                - padding_mask has shape [n_devices, n_batches, batch_size]

        Returns:
            Array of updated values for all states [n_devices, n_batches, batch_size]
        """
        actions, random_events, gamma, values = carry
        batched_states, padding_mask = batched_input

        def scan_fn(carry, batch_input):
            actions, random_events, gamma, current_values = carry
            batch, batch_padding_mask = batch_input

            # Process current batch using current_values from carry
            _, new_batch_values = self._calculate_updated_value_state_batch(
                (actions, random_events, gamma, current_values), batch
            )

            # Get indices for this batch - these map directly to positions in state space
            batch_indices = self._batch_get_indices(batch)

            # Update values in their original positions in state space
            # Only update non-padding states
            updated_values = current_values.at[batch_indices].set(
                jnp.where(
                    batch_padding_mask, current_values[batch_indices], new_batch_values
                )
            )

            return (actions, random_events, gamma, updated_values), new_batch_values

        # Process batches in the specified order
        if self.batch_order is not None:
            batched_states = batched_states[self.batch_order]
            padding_mask = padding_mask[self.batch_order]

        # Run scan with prefetching
        _, new_values = jax.lax.scan(scan_fn, carry, (batched_states, padding_mask))
        return new_values

    def _update_values(
        self,
        batched_states: BatchedStates,
        actions: ActionSpace,
        random_events: RandomEventSpace,
        gamma: float,
        values: ValueFunction,
    ) -> ValueFunction:
        """Update values for a batch of states using parallel processing.

        The semi-async nature comes from:
        1. Optionally shuffling states before processing
        2. Processing states in batches with immediate value updates
        3. Values are always stored in their correct positions

        Args:
            batched_states: Batched state vectors [n_devices, n_batches, batch_size, state_dim]
            actions: Action space [n_actions, action_dim]
            random_events: Random event space [n_events, event_dim]
            gamma: Discount factor
            values: Current value function [n_states]

        Returns:
            Array of updated values [n_states] in natural state order
        """
        # If using random strategy, shuffle states before processing
        shuffled_state_idxs = None
        padding_mask = None
        if self.config.shuffle_states:
            self.key, subkey = random.split(self.key)
            shuffled_state_idxs, batched_states, padding_mask = (
                self._jitted_shuffle_states(subkey)
            )
        else:
            # For fixed order, create padding mask for original batched states
            n_total = (
                batched_states.shape[0]
                * batched_states.shape[1]
                * batched_states.shape[2]
            )
            padding_mask = (jnp.arange(n_total) >= self.problem.n_states).reshape(
                batched_states.shape[0],  # n_devices
                batched_states.shape[1],  # n_batches
                batched_states.shape[2],  # batch_size
            )

        # Process batches semi-asynchronously
        padded_batched_values = self._calculate_updated_value_scan_state_batches_pmap(
            (actions, random_events, gamma, values), (batched_states, padding_mask)
        )

        # Unpad the values
        new_values = self._unbatch_results(padded_batched_values)

        # If states were shuffled, reorder values back to original state order
        if shuffled_state_idxs is not None:
            new_values = self._jitted_reorder_values(shuffled_state_idxs, new_values)

        return new_values

    def _iteration_step(self) -> tuple[ValueFunction, float]:
        """Run one iteration of semi-async value iteration.

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

        # Check convergence using selected test
        conv = self._convergence_test_fn(new_values, self.values)
        logger.info(
            f"Iteration {self.iteration}: {self._convergence_desc}: {conv:{self.convergence_format}}"
        )

        return new_values, conv

    def solve(self, max_iterations: int = 2000) -> SemiAsyncValueIterationState:
        """Run solver to convergence or max iterations.

        Args:
            max_iterations: Maximum number of iterations to run

        Returns:
            SolverState containing final values [n_states], optimal policy [n_states, action_dim],
            and SolverInfo including batch ordering info
        """
        for _ in range(max_iterations):
            self.iteration += 1
            new_values, conv = self._iteration_step()
            self.values = new_values

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

        logger.success("Semi-async value iteration completed")
        return self.solver_state

    @property
    def solver_state(self) -> SemiAsyncValueIterationState:
        """Get solver state for checkpointing."""
        return SemiAsyncValueIterationState(
            values=self.values,
            policy=self.policy,
            info=SemiAsyncValueIterationInfo(
                iteration=self.iteration,
                batch_order=self.batch_order,
            ),
        )

    def _restore_state_from_checkpoint(
        self, solver_state: SemiAsyncValueIterationState
    ) -> None:
        """Restore solver state from checkpoint."""
        self.values = solver_state.values
        self.policy = solver_state.policy
        self.iteration = solver_state.info.iteration
        self.batch_order = solver_state.info.batch_order

    def _compute_batch_assignments(
        self,
        n_batches: int,
        is_random: bool,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Determine batch processing order for semi-async updates.

        Args:
            n_batches: Number of batches to order
            is_random: Whether to use random ordering
            key: PRNG key for random ordering

        Returns:
            Tuple of (new order, new key)
        """

        def random_order(key):
            new_key, subkey = random.split(key)
            # Generate permutation for batch order
            order = random.permutation(subkey, jnp.arange(n_batches))
            # Compute inverse mapping: for each position in the reordered array,
            # store the original position that maps to it
            inverse = jnp.zeros_like(order)
            inverse = inverse.at[order].set(jnp.arange(len(order)))
            return (order, inverse), new_key

        def fixed_order(key):
            order = jnp.arange(n_batches)
            # For fixed order, inverse is same as order
            return (order, order), key

        return jax.lax.cond(
            is_random,
            random_order,
            fixed_order,
            key,
        )

    def _compute_inverse_order(self, order: jnp.ndarray) -> jnp.ndarray:
        """Compute inverse mapping for batch order.

        Args:
            order: Current batch order

        Returns:
            Inverse mapping for value lookups
        """
        inverse = jnp.zeros_like(order)
        return inverse.at[order].set(jnp.arange(len(order)))

    def _reorder_batches(self) -> None:
        """Update batch processing order for semi-async updates."""
        # Compute new batch order and its inverse in one step
        is_random = self.config.shuffle_states
        n_batches = self.batched_states.shape[1]
        (new_order, new_inverse), self.key = self._jitted_compute_batch_assignments(
            n_batches, is_random, self.key
        )
        self.batch_order = new_order
        self.inverse_order = new_inverse
