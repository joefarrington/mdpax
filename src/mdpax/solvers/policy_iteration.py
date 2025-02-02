"""Policy iteration solver for MDPs."""

import jax
import jax.numpy as jnp
from hydra.conf import dataclass
from jaxtyping import Array, Float
from loguru import logger

from mdpax.core.problem import Problem, ProblemConfig
from mdpax.core.solver import SolverConfig, SolverState
from mdpax.solvers.value_iteration import ValueIteration
from mdpax.utils.types import (
    ActionSpace,
    BatchedStates,
    Policy,
    RandomEventSpace,
    StateBatch,
    ValueFunction,
)


@dataclass
class PolicyIterationConfig(SolverConfig):
    """Configuration for the Policy Iteration solver.

    This solver performs policy iteration using parallel processing across devices.
    Each policy evaluation step uses batched computation for efficiency with large state spaces.

    Args:
        problem: Optional problem configuration. If not provided, can pass a Problem
            instance directly to the solver. If a Problem instance with a config is
            provided to the solver, its config will be extracted and stored here.
            Must be a ProblemConfig if provided.
        gamma: Discount factor in [0,1]
        epsilon: Convergence threshold for value changes during policy evaluation (must be positive)
        max_batch_size: Maximum states to process in parallel on each device (must be positive)
        jax_double_precision: Whether to use float64 precision
        verbose: Logging verbosity level (must be 0-4)
        checkpoint_dir: Directory to store checkpoints
        checkpoint_frequency: How often to save checkpoints (must be non-negative, 0 to disable)
        max_checkpoints: Maximum number of checkpoints to keep (must be non-negative)
        enable_async_checkpointing: Whether to save checkpoints asynchronously
        max_eval_iter: Maximum iterations for policy evaluation when using iterative method
        convergence_test: Strategy for testing convergence ("span" or "max_diff")
        reset_values_for_each_policy_eval: Whether to reset values to initial values at start of each policy evaluation
    """

    _target_: str = "mdpax.solvers.policy_iteration.PolicyIteration"
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
    max_eval_iter: int = 100
    convergence_test: str = "span"
    reset_values_for_each_policy_eval: bool = False

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
        if self.max_eval_iter <= 0:
            raise ValueError("max_eval_iter must be positive")
        if self.convergence_test not in ["span", "max_diff"]:
            raise ValueError("convergence_test must be 'span' or 'max_diff'")


class PolicyIteration(ValueIteration):
    """Policy iteration solver for MDPs.

    This solver implements policy iteration with parallel state updates across devices.
    States are automatically batched and padded for efficient parallel processing.

    The algorithm alternates between:
        1. Policy evaluation: computing values for current policy using iterative method with batched updates
        2. Policy improvement: one-step lookahead to find better policy

    The algorithm is considered to have converged when the policy does not change between successive
    iterations. For each iteration, the convergence of policy evaluation is tested using the span
    of differences in values between successive iterations by default (convergence_test='span').

    By default, the value estimates from the previous policy as used as the starting estimates
    for the next policy evaluation (reset_values_for_each_policy_eval=False). To start
    policy evaluation from the initial values in each iteration, set reset_values_for_each_policy_eval=True.

    To match the behaviour of pymdptoolbox's PolicyIteration class (with iterative evaluation)
    use the following arguments:
        - reset_values_for_each_policy_eval=True
        - convergence_test='max_diff'
        - max_eval_iter=10000
        - epsilon=1e-4

    Supports checkpointing for long-running problems using the CheckpointMixin.

    Args:
        problem: Problem instance or None if using config
        config: Configuration object. If provided, other kwargs are ignored.
        **kwargs: Parameters matching :class:`PolicyIterationConfig`.
            See Config class for detailed parameter descriptions.
    """

    Config = PolicyIterationConfig

    def __init__(
        self,
        problem: Problem | None = None,
        config: PolicyIterationConfig | None = None,
        **kwargs,
    ):
        """Initialize the solver."""
        super().__init__(problem=problem, config=config, **kwargs)

    def _setup_jax_functions(self) -> None:
        """Setup JAX functions for policy iteration."""
        super()._setup_jax_functions()
        self._calculate_policy_values_scan_state_batches_pmap = jax.pmap(
            self._calculate_policy_values_scan_state_batches,
            in_axes=((None, None, None, None, None), 0),
        )

        self._extract_policy_idx_scan_state_batches_pmap = jax.pmap(
            self._extract_policy_idx_scan_state_batches,
            in_axes=((None, None, None, None), 0),
        )

    def _initialize_solver_state_elements(self) -> None:
        """Initialize solver state elements."""
        # Set values to zero for policy initialization
        self.values = jnp.zeros(self.problem.n_states)
        self.policy = self._initialize_policy()
        self.values = self._initialize_values(self.batched_states)

        if self.config.reset_values_for_each_policy_eval:
            # store initial values for policy evaluation resets
            self.initial_values = self.values
        self.iteration = 0

    def _initialize_policy(self) -> Policy:
        """Initialize policy as custom policy or by maximizing immediate reward.

        If the problem provides an initial policy, use it. Otherwise, use the policy
        that maximizes immediate reward (using _extract_policy with zero values).
        """
        try:
            # Try to use problem's initial policy
            initial_policy = jax.vmap(self.problem.initial_policy)(
                self.problem.state_space
            )
        except NotImplementedError:

            # Extract policy using zero values (maximizes immediate reward)
            initial_policy = self._extract_policy()

        return initial_policy

    def _calculate_policy_value_state_batch(
        self,
        carry: tuple[ActionSpace, RandomEventSpace, float, ValueFunction, Policy],
        state_batch: StateBatch,
    ) -> tuple[tuple, Float[Array, "batch_size"]]:
        """Calculate values for a batch of states using their policy actions.

        Args:
            carry: Tuple of (actions, random_events, gamma, values, policy)
            state_batch: Batch of states to update [batch_size, state_dim]

        Returns:
            Tuple of (carry, new_values) where new_values has shape [batch_size]
        """
        actions, random_events, gamma, values, policy = carry

        # Get policy actions for this batch
        batch_indices = jax.vmap(self.problem.state_to_index)(state_batch)
        batch_actions = policy[batch_indices]  # Already contains action vectors

        # Calculate values using policy actions
        new_values = jax.vmap(
            self._calculate_updated_state_action_value,
            in_axes=(0, 0, None, None, None),
        )(state_batch, batch_actions, random_events, gamma, values)

        return carry, new_values

    def _calculate_policy_values_scan_state_batches(
        self,
        carry: tuple[ActionSpace, RandomEventSpace, float, ValueFunction, Policy],
        padded_batched_states: BatchedStates,
    ) -> Float[Array, "n_devices n_batches batch_size"]:
        """Calculate policy values for multiple batches of states.

        Uses jax.lax.scan to loop over batches efficiently.

        Args:
            carry: Tuple of (actions, random_events, gamma, values, policy)
            padded_batched_states: States prepared for batch processing
                Shape: [n_devices, n_batches, batch_size, state_dim]

        Returns:
            Array of updated values for all states [n_devices, n_batches, batch_size]
        """
        _, new_values = jax.lax.scan(
            self._calculate_policy_value_state_batch,
            carry,
            padded_batched_states,
        )
        return new_values

    def _calculate_policy_values(
        self,
        policy: Policy,
        values: ValueFunction,
    ) -> ValueFunction:
        """Calculate new values using only the actions specified by the policy.

        Uses batched computation and parallel processing across devices for efficiency
        with large state spaces.

        Args:
            policy: Current policy [n_states]
            values: Current values [n_states]

        Returns:
            New values [n_states]
        """
        # Process batches in parallel across devices
        padded_batched_values = self._calculate_policy_values_scan_state_batches_pmap(
            (
                self.problem.action_space,
                self.problem.random_event_space,
                self.gamma,
                values,
                policy,
            ),
            self.batched_states,
        )

        # Unbatch and remove padding
        new_values = self._unbatch_results(padded_batched_values)
        new_values = new_values.reshape(-1)
        return new_values

    def _evaluate_policy(
        self,
        policy: Policy,
        starting_values: ValueFunction | None = None,
    ) -> ValueFunction:
        """Evaluate policy using iterative updates with batched computation.

        Similar to value iteration but only considers the current policy's action
        for each state. Uses batched updates for efficiency with large state spaces.
        Uses same convergence test and threshold as main iteration.

        Args:
            policy: Current policy to evaluate [n_states]
            starting_values: Initial values to start from [n_states], uses zero values if None

        Returns:
            Updated values for the policy [n_states]
        """
        # If no starting values provided, use either zero values or current values based on config
        if starting_values is None:
            values = (
                self.initial_values
                if self.config.reset_values_for_each_policy_eval
                else self.values
            )
        else:
            values = starting_values

        # Iterate until values converge or max iterations reached
        for eval_iter in range(self.config.max_eval_iter):
            # Calculate new values using only the policy's actions
            new_values = self._calculate_policy_values(policy, values)

            # Check convergence using same test as main iteration
            conv = self._convergence_test_fn(new_values, values)
            if self.verbose:
                logger.debug(
                    f"Policy evaluation iteration {eval_iter+1}: "
                    f"{self._convergence_desc}: {conv:{self.convergence_format}}"
                )

            if conv < self.conv_threshold:
                break

            values = new_values

        return values

    def _iteration_step(self) -> tuple[Policy, int]:
        """Perform one iteration of policy iteration.

        1. Evaluate current policy to get values (starting values determined by config)
        2. Improve policy using one-step lookahead

        Returns:
            Tuple of (new_policy, n_changed) where:
                new_policy: Improved policy [n_states]
                n_changed: Number of states where policy changed
        """
        # Evaluate current policy (starting point determined by config)
        self.values = self._evaluate_policy(self.policy)

        # Improve policy using parent's policy extraction
        new_policy = self._extract_policy()

        # Count number of states where policy differs (comparing full action vectors)
        n_changed = jnp.any(new_policy != self.policy, axis=1).sum()

        return new_policy, n_changed

    def solve(self, max_iterations: int = 1000) -> SolverState:
        """Run solver to convergence or max iterations.

        Policy iteration is guaranteed to converge in finite iterations for discounted MDPs.
        Stops when policy stops changing or max iterations reached.

        Args:
            max_iterations: Maximum number of iterations to run

        Returns:
            SolverState containing final values [n_states], optimal policy [n_states],
            and SolverInfo including iteration count
        """
        for _ in range(max_iterations):
            self.iteration += 1

            # Do one iteration of policy iteration
            new_policy, n_changed = self._iteration_step()
            self.policy = new_policy

            # Log progress
            logger.info(
                f"Iteration {self.iteration}: Policy updated for {n_changed} state(s) ({n_changed/self.problem.n_states*100:.2f}%)"
            )

            # Check for convergence
            if n_changed == 0:
                logger.info(f"Policy converged at iteration {self.iteration}")
                break

            # Save checkpoint if enabled
            if (
                self.is_checkpointing_enabled
                and self.iteration % self.checkpoint_frequency == 0
            ):
                self.save(self.iteration)

        if n_changed > 0:
            logger.info("Maximum iterations reached")

        # Final checkpoint if enabled
        if self.is_checkpointing_enabled:
            self.save(self.iteration)

        logger.success("Policy iteration completed")
        return self.solver_state
