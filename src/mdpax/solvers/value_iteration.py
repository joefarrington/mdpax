"""Value iteration solver for MDPs."""

from pathlib import Path
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from hydra.conf import dataclass
from loguru import logger

from mdpax.core.problem import Problem
from mdpax.core.solver import (
    Solver,
    SolverState,
    SolverWithCheckpointConfig,
)
from mdpax.utils.checkpointing import CheckpointMixin


@dataclass
class ValueIterationConfig(SolverWithCheckpointConfig):
    """Configuration for the Value Iteration solver.

    This is the base value iteration solver that performs synchronous updates
    over all states. It inherits all parameters from the base SolverConfig.
    """

    _target_: str = "mdpax.solvers.value_iteration.ValueIteration"


class ValueIteration(Solver, CheckpointMixin):
    """Value iteration solver for MDPs."""

    def __init__(
        self,
        problem: Problem,
        gamma: float = 0.99,
        max_iter: int = 1000,
        epsilon: float = 1e-3,
        batch_size: int = 1024,
        jax_double_precision: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_frequency: int = 0,
        max_checkpoints: int = 1,
        enable_async_checkpointing: bool = True,
        verbose: int = 2,
    ):
        """Initialize the solver."""
        super().__init__(
            problem, gamma, max_iter, epsilon, batch_size, jax_double_precision, verbose
        )
        self.setup_checkpointing(
            checkpoint_dir,
            checkpoint_frequency,
            max_checkpoints=max_checkpoints,
            enable_async_checkpointing=enable_async_checkpointing,
        )
        self.policy = None

    def _setup_solver(self) -> None:
        """Setup solver-specific computations.

        This method sets up the vectorized computation of state-action values
        and the multi-device batch processing for value updates and policy extraction.
        Wrapping these in pmap allows for efficient parallel processing across devices
        and JIT compilation is automatically handled.
        """

        self._transition_vmap_random_event = jax.vmap(
            self.problem.transition, in_axes=(None, None, 0)
        )

        self._compute_states_actions_values_vmap_states_actions = jax.vmap(
            jax.vmap(
                self._compute_state_action_value, in_axes=(None, 0, None, None)
            ),  # Over actions
            in_axes=(0, None, None, None),  # Over states
        )

        self._compute_values_scan_batches_pmap_state_batches = jax.pmap(
            self._compute_values_scan_batches,
            in_axes=(0, None, None, None),
            axis_name="device",
        )

        self._extract_policy_idxs_scan_batches_pmap_state_batches = jax.pmap(
            self._extract_policy_idxs_scan_batches,
            in_axes=(0, None, None, None),
            axis_name="device",
        )

    def _compute_state_action_value(
        self, state: jnp.ndarray, action: jnp.ndarray, values: jnp.ndarray, gamma: float
    ) -> float:
        """Compute expected value for a state-action pair.

        Args:
            state: Current state [state_dim]
            action: Action to evaluate [action_dim]
            values: Current value estimates [n_states]

        Returns:
            Expected value of taking action in state
        """
        # Get transition probabilities for this state-action pair
        probs = self.problem.random_event_probabilities(state, action)

        # Compute next states and rewards for all possible random events
        next_states, rewards = self._transition_vmap_random_event(
            state, action, self.problem.random_event_space
        )

        # Convert next states to indices for value lookup
        next_state_indices = jax.vmap(self.problem.state_to_index)(next_states)

        # Get next state values
        next_values = values[next_state_indices]

        # Compute expected value
        return (rewards + gamma * next_values).dot(probs)

    def _compute_values_for_batch(
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        values: jnp.ndarray,
        gamma: float,
    ) -> jnp.ndarray:
        """Process a batch of states to compute new values.

        This method:
        1. Gets the action space from the problem
        2. Uses _compute_state_action_values to compute values for all
           state-action pairs
        3. Takes the maximum over actions to get the new value for each state

        Args:
            states: Batch of states to process [batch_size, state_dim]
            values: Current value estimates [n_states]
            gamma: Discount factor
        Returns:
            New values for the batch of states [batch_size]
        """
        # Compute state-action values for all states and actions in batch
        state_action_values = self._compute_states_actions_values_vmap_states_actions(
            states, actions, values, gamma
        )
        # Take maximum over actions
        return jnp.max(state_action_values, axis=1)

    def _compute_values_scan_batches(
        self,
        batched_states: jnp.ndarray,
        actions: jnp.ndarray,
        values: jnp.ndarray,
        gamma: float,
    ) -> jnp.ndarray:
        """Process batches of states"""

        def scan_fn(_, batch):
            return (None, self._compute_values_for_batch(batch, actions, values, gamma))

        _, new_values = jax.lax.scan(scan_fn, None, batched_states)
        return new_values

    def _extract_policy_idxs_for_batch(
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        values: jnp.ndarray,
        gamma: float,
    ) -> jnp.ndarray:
        """Extract policy for a batch of states."""
        # Compute state-action values
        state_action_values = self._compute_states_actions_values_vmap_states_actions(
            states, actions, values, gamma
        )
        # Select best action idxs
        return jnp.argmax(state_action_values, axis=1)

    def _extract_policy_idxs_scan_batches(
        self,
        batched_states: jnp.ndarray,
        actions: jnp.ndarray,
        values: jnp.ndarray,
        gamma: float,
    ) -> jnp.ndarray:
        """Process batches of states"""

        def scan_fn(_, batch):
            return (
                None,
                self._extract_policy_idxs_for_batch(batch, actions, values, gamma),
            )

        _, new_policy = jax.lax.scan(scan_fn, None, batched_states)
        return new_policy

    def _iteration_step(self) -> Tuple[jnp.ndarray, float]:
        """Perform one iteration step.

        Returns:
            Tuple of (new values, convergence measure)
        """
        # Process all batches
        new_values = self._update_values(
            self.batched_states, self.problem.action_space, self.values, self.gamma
        )
        # Compute convergence measure
        conv = jnp.max(jnp.abs(new_values - self.values))
        return new_values, conv

    def _update_values(
        self,
        batched_states: jnp.ndarray,
        actions: jnp.ndarray,
        values: jnp.ndarray,
        gamma: float,
    ) -> jnp.ndarray:
        """Update values for a batch of states."""
        device_values = self._compute_values_scan_batches_pmap_state_batches(
            batched_states, actions, values, gamma
        )
        new_values = jnp.reshape(device_values, (-1,))
        # Remove padding if needed
        new_values = self._unpad_results(new_values)
        return new_values

    def _extract_policy(self, values: jnp.ndarray) -> jnp.ndarray:
        """Extract policy from values."""

        # Multi-device policy extraction
        policy_action_idxs = self._extract_policy_idxs_scan_batches_pmap_state_batches(
            self.batched_states, self.problem.action_space, values, self.gamma
        ).reshape(-1)

        # Remove padding if needed
        policy_action_idxs = self._unpad_results(policy_action_idxs)

        # Look up actual actions from policy
        self.policy = jnp.take(self.problem.action_space, policy_action_idxs, axis=0)
        return self.policy

    def solve(self) -> SolverState:
        """Run solver to convergence.

        Returns:
            Tuple of (optimal values, optimal policy)
        """

        while self.iteration < self.max_iter:
            self.iteration += 1

            # Perform iteration step
            new_values, conv = self._iteration_step()

            # Update values and iteration count
            self.values = new_values

            logger.info(f"Iteration {self.iteration} maximum delta: {conv:.4f}")

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

        # Save checkpoint if enabled
        if self.is_checkpointing_enabled:
            self.save(self.iteration)

        # Extract policy if converged or on final iteration
        logger.info("Extracting policy")
        self.policy = self._extract_policy(self.values)
        logger.info("Policy extracted")

        logger.success("Value iteration completed")
        return self.solver_state

    def _get_solver_config(self) -> ValueIterationConfig:
        """Get solver configuration for reconstruction."""
        return ValueIterationConfig(
            problem=self.problem.get_problem_config(),
            gamma=self.gamma,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            batch_size=self.batch_size,
            jax_double_precision=self.jax_double_precision,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_frequency=self.checkpoint_frequency,
            max_checkpoints=self.max_checkpoints,
            enable_async_checkpointing=self.enable_async_checkpointing,
        )

    def _restore_state_from_checkpoint(self, solver_state: SolverState) -> None:
        """Restore solver state from checkpoint."""
        self.values = solver_state.values
        self.policy = solver_state.policy
        self.iteration = solver_state.info.iteration
