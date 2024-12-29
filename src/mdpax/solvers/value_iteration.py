"""Value iteration solver for MDPs."""

from pathlib import Path
from typing import Optional, Tuple, Union

import chex
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
        # Cache problem methods at the start

        self._calculate_updated_value_scan_state_batches_pmap = jax.pmap(
            self._calculate_updated_value_scan_state_batches,
            in_axes=((None, None, None, None), 0),
        )

        self._extract_policy_idx_scan_state_batches_pmap = jax.pmap(
            self._extract_policy_idx_scan_state_batches,
            in_axes=((None, None, None, None), 0),
        )

    def _get_value_next_state(
        self, next_state: chex.Array, values: chex.Array
    ) -> float:
        """Lookup the value of the next state in the value function from the
        previous iteration."""
        return values[self.problem.state_to_index(next_state)]

    def _calculate_updated_state_action_value(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        random_events: chex.Array,
        gamma: float,
        values: chex.Array,
    ) -> float:
        """Update the state-action value for a given state, action pair"""
        (
            next_states,
            single_step_rewards,
        ) = jax.vmap(
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
        new_state_action_value = (single_step_rewards + gamma * next_state_values).dot(
            probs
        )
        return new_state_action_value

    def _calculate_updated_value(
        self,
        state: chex.Array,
        actions: Union[int, chex.Array],
        random_events: chex.Array,
        gamma: float,
        values: chex.Array,
    ) -> float:
        """Update the value for a given state, by taking the max of the
          updated state-action
        values over all actions"""
        return jnp.max(
            jax.vmap(
                self._calculate_updated_state_action_value,
                in_axes=(None, 0, None, None, None),
            )(state, actions, random_events, gamma, values)
        )

    def _calculate_updated_value_state_batch(
        self, carry, state_batch: chex.Array
    ) -> Tuple[Tuple[Union[int, chex.Array], chex.Array, chex.Array], chex.Array]:
        """Calculate the updated value for a batch of states"""
        actions, random_events, gamma, values = carry
        new_values = jax.vmap(
            self._calculate_updated_value, in_axes=(0, None, None, None, None)
        )(state_batch, actions, random_events, gamma, values)
        return carry, new_values

    def _calculate_updated_value_scan_state_batches(
        self,
        carry: Tuple[Union[int, chex.Array], chex.Array, chex.Array],
        padded_batched_states: chex.Array,
    ) -> chex.Array:
        """Calculate the updated value for multiple batches of states, using
        jax.lax.scan to loop over batches of states."""
        carry, new_values_padded = jax.lax.scan(
            self._calculate_updated_value_state_batch,
            carry,
            padded_batched_states,
        )
        return new_values_padded

    def _extract_policy_idx_one_state(
        self,
        state: chex.Array,
        actions: Union[int, chex.Array],
        random_events: chex.Array,
        gamma: float,
        values: chex.Array,
    ) -> int:
        """Extract the best action for a single state, by taking the argmax of the
        updated state-action values over all actions"""
        best_action_idx = jnp.argmax(
            jax.vmap(
                self._calculate_updated_state_action_value,
                in_axes=(None, 0, None, None, None),
            )(state, actions, random_events, gamma, values)
        )
        return best_action_idx

    def _extract_policy_idx_state_batch(
        self,
        carry: Tuple[Union[int, chex.Array], chex.Array, chex.Array],
        state_batch: chex.Array,
    ) -> chex.Array:
        """Extract the best action for a batch of states"""
        actions, random_events, gamma, values = carry
        best_action_idxs = jax.vmap(
            self._extract_policy_idx_one_state, in_axes=(0, None, None, None, None)
        )(state_batch, actions, random_events, gamma, values)
        return carry, best_action_idxs

    def _extract_policy_idx_scan_state_batches(
        self,
        carry: Tuple[Union[int, chex.Array], chex.Array, chex.Array],
        padded_batched_states: chex.Array,
    ) -> chex.Array:
        """Extract the best action for multiple batches of states, using jax.lax.scan
        o loop over batches of states."""
        carry, best_action_idxs_padded = jax.lax.scan(
            self._extract_policy_idx_state_batch,
            carry,
            padded_batched_states,
        )
        return best_action_idxs_padded

    def _iteration_step(self) -> Tuple[jnp.ndarray, float]:
        """Perform one iteration step.

        Returns:
            Tuple of (new values, convergence measure)
        """
        logger.debug(
            f"Starting iteration step with values shape {self.values.shape} "
            f"and batched_states shape {self.batched_states.shape}"
        )
        new_values = self._update_values(
            self.batched_states,  # Shape could vary
            self.problem.action_space,
            self.problem.random_event_space,
            self.gamma,
            self.values,
        )
        # Compute convergence measure
        conv = jnp.max(jnp.abs(new_values - self.values))
        logger.debug("Completed iteration step")
        return new_values, conv

    def _update_values(
        self,
        batched_states: jnp.ndarray,
        actions: jnp.ndarray,
        random_events: jnp.ndarray,
        gamma: float,
        values: jnp.ndarray,
    ) -> jnp.ndarray:
        """Update values for a batch of states."""
        logger.debug("Starting update_values")
        padded_batched_values = self._calculate_updated_value_scan_state_batches_pmap(
            (actions, random_events, gamma, values), batched_states
        )
        logger.debug("Computed padded batched values")
        padded_values = jnp.reshape(padded_batched_values, (-1,))
        logger.debug("Reshaped padded values")
        # Remove padding if needed
        new_values = self._unpad_results(padded_values)
        logger.debug("Unpadded values")
        logger.debug("Finished update_values")
        return new_values

    def _extract_policy(
        self,
    ) -> jnp.ndarray:
        """Extract policy from values."""

        # Multi-device policy extraction
        padded_batched_policy_action_idxs = (
            self._extract_policy_idx_scan_state_batches_pmap(
                (
                    self.problem.action_space,
                    self.problem.random_event_space,
                    self.gamma,
                    self.values,
                ),
                self.batched_states,
            )
        )

        padded_policy_action_idxs = jnp.reshape(
            padded_batched_policy_action_idxs, (-1,)
        )

        # Remove padding if needed
        policy_action_idxs = self._unpad_results(padded_policy_action_idxs)

        # Look up actual actions from policy
        return jnp.take(self.problem.action_space, policy_action_idxs, axis=0)

    def solve(self) -> SolverState:
        """Run solver to convergence.

        Returns:
            Tuple of (optimal values, optimal policy)
        """

        while self.iteration < self.max_iter:
            self.iteration += 1
            logger.debug(f"Values shape: {self.values.shape}")
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
        self.policy = self._extract_policy()
        logger.info("Policy extracted")

        logger.success("Value iteration completed")
        return self.solver_state

    def _get_solver_config(self) -> ValueIterationConfig:
        """Get solver configuration for reconstruction."""
        return ValueIterationConfig(
            problem=self.problem.get_problem_config(),
            gamma=float(self.gamma),
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
