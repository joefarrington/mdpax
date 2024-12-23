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
            next_states, rewards = jax.vmap(
                lambda e: self.problem.transition(state, action, e)
            )(self.problem.random_event_space)

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

    def _extract_policy(self, values: jnp.ndarray) -> jnp.ndarray:
        """Extract policy from values."""
        if self.n_devices > 1:
            # Multi-device policy extraction
            device_policies = self._extract_batch_policy_pmap(
                self.batched_states, values
            )
            # Combine results from all devices
            self.policy = jnp.reshape(device_policies, (-1,))
        else:
            # Single device policy extraction
            policy_batches = []
            for batch in self.batched_states:
                policy_batch = self._extract_batch_policy(batch, values)
                policy_batches.append(policy_batch)
            self.policy = jnp.concatenate(policy_batches)

        # Remove padding if needed
        if self.n_pad > 0:
            self.policy = self.policy[: -self.n_pad]

        # Look up actual actions from policy
        self.policy = jnp.take(self.problem.action_space, self.policy, axis=0)
        return self.policy

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
