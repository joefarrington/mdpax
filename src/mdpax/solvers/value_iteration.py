"""Value iteration solver with efficient parallel processing and batching."""

from pathlib import Path
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp

from mdpax.core.problem import Problem
from mdpax.core.solver import Solver
from mdpax.utils.checkpointing import CheckpointMixin


class ValueIteration(Solver, CheckpointMixin):
    """Value iteration solver using parallel processing.

    This solver implements parallel value iteration with the following features:
    1. Automatic batching of states for efficient device utilization
    2. Multi-device parallel processing using pmap
    3. JIT compilation of core computations
    4. Optional checkpointing for large problems
    5. Efficient state-action value computation
    6. Batched policy extraction

    The solver expects the Problem instance to provide:
    - state_space: Array of states [n_states, state_dim]
    - action_space: Array of actions [n_actions, action_dim]
    - random_event_space: Array of random events [n_events, event_dim]
    - transition: Function mapping (state, action, random_event) to (next_state, reward)
    - random_event_probabilities: Function mapping (state, action) to event probs
    - state_to_index: Function mapping state vector to index
    """

    def __init__(
        self,
        problem: Problem,
        gamma: float = 0.99,
        max_iter: int = 1000,
        epsilon: float = 0.01,
        batch_size: int = 1024,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_frequency: int = 0,
        max_checkpoints: int = 1,
        enable_async_checkpointing: bool = False,
    ):
        """Initialize the solver.

        Args:
            problem: MDP problem to solve (must be a Problem instance)
            gamma: Discount factor in [0,1]
            max_iter: Maximum iterations (positive integer)
            epsilon: Convergence threshold (positive float)
            batch_size: Size of state batches (positive integer)
            checkpoint_dir: Directory to save checkpoints (optional)
            checkpoint_frequency: Frequency of checkpoints
                (positive integer, 0 to disable)
            max_checkpoints: Maximum number of checkpoints to keep (positive integer)
            cleanup_on_completion: Whether to cleanup checkpoints on completion (bool)
            enable_async_checkpointing: Whether to enable asynchronous
                checkpointing (bool)

        Raises:
            TypeError: If problem is not a Problem instance
            ValueError: If parameters are out of valid ranges
        """
        if not isinstance(problem, Problem):
            raise TypeError("problem must be an instance of Problem")

        super().__init__(problem, gamma, max_iter, epsilon, batch_size)
        self.setup_checkpointing(
            checkpoint_dir,
            checkpoint_frequency,
            max_checkpoints=max_checkpoints,
            enable_async=enable_async_checkpointing,
        )
        self.policy = None

    def _setup(self) -> None:
        """Setup solver-specific computations."""
        # JIT compile core computations for single device
        self._compute_state_action_values = jax.jit(
            self._batch_state_action_values, static_argnums=(0,)
        )

        # Setup multi-device batch processing
        if self.n_devices > 1:
            self._process_batches = jax.pmap(
                self._scan_batches,
                in_axes=((None, None), 0),  # (values, gamma), batched_states
                axis_name="device",
            )
            self._extract_batch_policy_pmap = jax.pmap(
                self._extract_policy_for_batch,
                in_axes=(0, None),  # states, values
                axis_name="device",
            )
        else:
            # Single device processing
            self._process_batches = self._setup_single_device()
            self._extract_batch_policy = jax.jit(
                self._extract_policy_for_batch,
            )

    def _batch_state_action_values(
        self, states: jnp.ndarray, values: jnp.ndarray, *args
    ) -> jnp.ndarray:
        """Compute state-action values for a batch of states."""

        # Vectorize over states and actions
        def compute_action_value(state, action):
            # Get transition probabilities for this state-action pair
            probs = self.problem.random_event_probabilities(state, action)

            # Compute next states and rewards for all possible random events
            transitions = jax.vmap(lambda e: self.problem.transition(state, action, e))(
                self.problem.random_event_space
            )

            # Unpack next states and rewards
            next_states, rewards = jax.tree.map(lambda x: jnp.array(x), transitions)

            # Convert next states to indices for value lookup
            next_state_indices = jax.vmap(self.problem.state_to_index)(next_states)

            # Get next state values
            next_values = values[next_state_indices]

            # Compute expected value
            return (rewards + self.gamma * next_values).dot(probs)

        # Vectorize over states and actions
        return jax.vmap(
            jax.vmap(compute_action_value, in_axes=(None, 0)),  # Over actions
            in_axes=(0, None),  # Over states
        )(states, self.problem.action_space)

    def _batch_value_calculation(
        self, states: jnp.ndarray, values: jnp.ndarray, *args
    ) -> jnp.ndarray:
        """Process a batch of states."""
        # Compute state-action values
        state_action_values = self._compute_state_action_values(states, values)

        # Take maximum over actions
        return jnp.max(state_action_values, axis=1)

    def _extract_policy_for_batch(
        self, states: jnp.ndarray, values: jnp.ndarray
    ) -> jnp.ndarray:
        """Extract policy for a batch of states."""
        # Compute state-action values
        state_action_values = self._compute_state_action_values(states, values)

        # Select best actions
        return jnp.argmax(state_action_values, axis=1)

    def _scan_batches(
        self,
        carry: Tuple[jnp.ndarray, float],  # (values, gamma)
        batched_states: jnp.ndarray,
    ) -> jnp.ndarray:
        """Process batches of states (for multi-device case)."""
        values, gamma = carry

        def scan_fn(_, batch):
            return (None, self._batch_value_calculation(batch, values, gamma))

        _, new_values = jax.lax.scan(scan_fn, None, batched_states)
        return new_values

    def _setup_single_device(self):
        """Setup processing for single device."""
        batch_fn = jax.jit(self._batch_value_calculation)

        def process_batches(carry, states):
            def scan_fn(_, batch):
                return (None, batch_fn(batch, *carry))

            _, new_values = jax.lax.scan(scan_fn, None, states)
            return new_values

        return jax.jit(process_batches)

    def _iteration_step(self) -> Tuple[jnp.ndarray, float]:
        """Perform one iteration step.

        Returns:
            Tuple of (new values, convergence measure)
        """
        # Process all batches
        device_values = self._process_batches(
            (self.values, self.gamma), self.batched_states
        )
        # Combine results from all devices
        new_values = jnp.reshape(device_values, (-1,))

        # Remove padding if needed
        if self.n_pad > 0:
            new_values = new_values[: -self.n_pad]

        # Compute convergence measure
        conv = jnp.max(jnp.abs(new_values - self.values))

        return new_values, conv

    def solve(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Run solver to convergence.

        Returns:
            Tuple of (optimal values, optimal policy)
        """
        while self.iteration < self.max_iter:
            # Perform iteration step
            new_values, conv = self._iteration_step()

            # Update values and iteration count
            self.values = new_values
            self.iteration += 1

            # Check convergence
            if conv < self.epsilon:
                break

            # Save checkpoint if enabled
            if self.is_checkpointing_enabled:
                self.save_checkpoint(self.iteration)

            # Extract policy if converged or on final iteration
        if self.n_devices > 1:
            # Multi-device policy extraction
            device_policies = self._extract_batch_policy_pmap(
                self.batched_states, new_values
            )
            # Combine results from all devices
            self.policy = jnp.reshape(device_policies, (-1,))
        else:
            # Single device policy extraction
            policy_batches = []
            for batch in self.batched_states:
                policy_batch = self._extract_batch_policy(batch, new_values)
                policy_batches.append(policy_batch)
            self.policy = jnp.concatenate(policy_batches)

        # Remove padding if needed
        if self.n_pad > 0:
            self.policy = self.policy[: -self.n_pad]

        # Look up actual actions from policy
        self.policy = jnp.take(self.problem.action_space, self.policy, axis=0)

        return self.values, self.policy

    def _get_checkpoint_state(self) -> dict:
        """Get solver state for checkpointing.

        Returns:
            dict containing all necessary state to resume solving:
            - values: Current value function
            - policy: Current policy (if exists)
            - iteration: Current iteration count
        """
        cp_state = {
            "values": self.values,
            "iteration": self.iteration,
        }
        if self.policy is not None:
            cp_state["policy"] = self.policy
        return cp_state

    def _restore_from_checkpoint(self, cp_state: dict) -> None:
        """Restore solver state from checkpoint.

        Args:
            cp_state: State dict from get_checkpoint_state()
        """
        self.values = cp_state["values"]
        self.iteration = cp_state["iteration"]
        if "policy" in cp_state:
            self.policy = cp_state["policy"]
