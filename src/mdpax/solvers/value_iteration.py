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
        self.gamma = jnp.array(gamma)
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
        transition_fn = self.problem.transition
        prob_fn = jax.vmap(
            self.problem.random_event_probability, in_axes=(None, None, 0)
        )  # vmap over random events
        state_to_idx_fn = self.problem.state_to_index

        def get_value_next_state(
            next_state: chex.Array, values: chex.Array, state_to_idx_fn=state_to_idx_fn
        ) -> float:
            """Lookup value of next state in value function from previous iteration."""
            return values[state_to_idx_fn(next_state)]

        # 1. Base state-action value computation
        def compute_state_action_value(
            state,
            action,
            random_events,
            values,
            gamma,
            transition_fn=transition_fn,
            prob_fn=prob_fn,
            get_value_next_state_fn=get_value_next_state,
        ):
            probs = prob_fn(state, action, random_events)
            next_states, rewards = jax.vmap(
                transition_fn,
                in_axes=(None, None, 0),
            )(state, action, random_events)
            next_values = jax.vmap(get_value_next_state, in_axes=(0, None))(
                next_states, values
            )
            q = (rewards + gamma * next_values).dot(probs)
            return q

        # Compile state-action value computation (vmap over actions)
        self._compute_state_action_values = jax.jit(
            jax.vmap(compute_state_action_value, in_axes=(None, 0, None, None, None))
        )

        # 2. Value computation for batches of states
        def compute_batch_values(states, actions, random_events, values, gamma):
            state_action_values = self._compute_state_action_values(
                states, actions, random_events, values, gamma
            )
            return jnp.max(state_action_values)

        # Compile batch value computation (vmap over states)
        self._compute_batch_values = jax.jit(
            jax.vmap(compute_batch_values, in_axes=(0, None, None, None, None))
        )

        # 3. Multi-device value computation
        def process_value_device_batch(
            batched_states, actions, random_events, values, gamma
        ):
            def batch_fn(carry, state_batch):
                return (carry, self._compute_batch_values(state_batch, *carry))

            values = jax.lax.scan(
                batch_fn, (actions, random_events, values, gamma), batched_states
            )[1]
            return values

        # Compile multi-device value computation (pmap includes jit)
        self._compute_values_state_batches = jax.pmap(
            process_value_device_batch,
            in_axes=(0, None, None, None, None),  # No more static_broadcasted_argnums
        )

        # 4. Policy extraction for batches of states
        def extract_batch_policy_idxs(states, actions, random_events, values, gamma):
            state_action_values = self._compute_state_action_values(
                states, actions, random_events, values, gamma
            )
            return jnp.argmax(state_action_values)

        # Compile batch policy computation (vmap over states)
        self._extract_batch_policy_idxs = jax.jit(
            jax.vmap(extract_batch_policy_idxs, in_axes=(0, None, None, None, None))
        )

        # 5. Multi-device policy computation
        def process_policy_idxs_device_batch(
            batched_states, actions, random_events, values, gamma
        ):
            def batch_fn(carry, state_batch):
                return (carry, self._extract_batch_policy_idxs(state_batch, *carry))

            return jax.lax.scan(
                batch_fn, (actions, random_events, values, gamma), batched_states
            )[1]

        # Compile multi-device policy computation (pmap includes jit)
        self._extract_policy_idxs_state_batches = jax.pmap(
            process_policy_idxs_device_batch,
            in_axes=(0, None, None, None, None),  # No static args
        )

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
            self.values,
            self.gamma,
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
        values: jnp.ndarray,
        gamma: float,
    ) -> jnp.ndarray:
        """Update values for a batch of states."""
        logger.debug("Starting update_values")
        padded_batched_values = self._compute_values_state_batches(
            batched_states, actions, random_events, values, gamma
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
        padded_batched_policy_action_idxs = self._extract_policy_idxs_state_batches(
            self.batched_states,
            self.problem.action_space,
            self.problem.random_event_space,
            self.values,
            self.gamma,
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
