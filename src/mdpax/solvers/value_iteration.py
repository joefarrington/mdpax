import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import re
import logging
import math
from pathlib import Path
from typing import Union, Tuple, Dict, List, Optional
import chex
from datetime import datetime
from mdpax.core.problem import Problem

import jax
jax.config.update("jax_enable_x64", True)

# Enable logging
log = logging.getLogger("ValueIterationRunner")


class ValueIterationRunner:
    def __init__(
        self,
        problem: Problem,
        max_batch_size: int,
        epsilon: float,
        gamma: float,
        output_directory: Optional[Union[str, Path]] = None,
        checkpoint_frequency: int = 0,
        resume_from_checkpoint: Union[bool, str] = False,
    ):
        """Base class for running value iteration

        Args:
            max_batch_size: Maximum number of states to update in parallel using vmap, will depend on GPU memory
            epsilon: Convergence criterion for value iteration
            gamma: Discount factor
            output_directory: Directory to save output to, if None, will create a new directory
            checkpoint_frequency: Frequency with which to save checkpoints, 0 for no checkpoints
            resume_from_checkpoint: If False, start from scratch; if filename, resume from checkpoint

        """
        
        self.problem = problem
        self.max_batch_size = max_batch_size
        self.epsilon = epsilon
        self.gamma = gamma

        if output_directory is None:
            now = datetime.now()
            date = now.strftime("%Y-%m-%d)")
            time = now.strftime("%H-%M-%S")
            self.output_directory = Path(f"vi_output/{date}/{time}").absolute()
        else:
            self.output_directory = Path(output_directory).absolute()

        self.checkpoint_frequency = checkpoint_frequency

        self.checkpoint_frequency = checkpoint_frequency
        if self.checkpoint_frequency > 0:
            self.cp_path = self.output_directory / "checkpoints"
            self.cp_path.mkdir(parents=True, exist_ok=True)

        self.resume_from_checkpoint = resume_from_checkpoint
        self._setup()

    ### Optional methods to implement in subclass ###

    def _setup_before_states_actions_random_outcomes_created(self) -> None:
        """Function that is run during setup before the arrays of states, actions
        and random outcomes are created. Use this to perform any calculations or set
        any properties that are required for those arrays to be created."""
        pass

    def _setup_after_states_actions_random_outcomes_created(self) -> None:
        """Function that is run during setup after the arrays of states, actions
        and random outcomes are created. Use this to perform any calculations or set
        any properties that depend on those arrays having been created."""
        pass

    ### End of optional methods to implement in subclass ###

    def run_value_iteration(
        self,
        max_iter: int = 10000,
        min_iter: int = 1,
        extract_policy: bool = True,
    ) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """Run value iteration for a given number of iterations, or until convergence. Optionally save checkpoints of the

        Args:
            value function after each iteration, and the final value function and policy at the end of the run.
            max_iter: maximum number of iterations to run
            min_iter: minimum number of iterations to run, even if convergence is reached before this
            extract_policy: whether to save the final policy as a csv file

        Returns:
            A dictionary containing information to log, the final value function and, optionally, the policy

        """

        # If already run more than max_iter, raise an error
        if self.iteration > max_iter:
            raise ValueError(
                f"At least {max_iter} iterations have already been completed"
            )

        # If min_iter greater than max_iter, raise an error
        if min_iter > max_iter:
            raise ValueError(f"min_iter must be less than or equal to max_iter")

        log.info(f"Starting value iteration at iteration {self.iteration}")

        for i in range(self.iteration, max_iter + 1):
            padded_batched_V = self.calculate_updated_value_scan_state_batches_pmap(
                (self.actions, self.possible_random_outcomes, self.V_old),
                self.padded_batched_states,
            )

            V = self._unpad(padded_batched_V.reshape(-1), self.n_pad)

            # Check for convergence
            if self.check_converged(i, min_iter, V, self.V_old):
                break
            else:

                if self.checkpoint_frequency > 0 and (
                    i % self.checkpoint_frequency == 0
                ):
                    values_df = pd.DataFrame(V, index=self.states, columns=["V"])
                    values_df.to_csv(self.cp_path / f"values_{i}.csv")

                self.V_old = V

            self.iteration += 1

        to_return = {}

        # Put final values into pd.DataFrame to return
        values_df = pd.DataFrame(np.array(V), index=self.state_tuples, columns=["V"])
        to_return[f"V"] = values_df

        # If extract_policy is True, extract policy with one-step ahead search
        # to return in output
        if extract_policy:
            log.info("Extracting final policy")
            best_order_actions_df = self.get_policy(V)
            log.info("Final policy extracted")
            to_return["policy"] = best_order_actions_df

        to_return["output_info"] = self.output_info

        self.V = V.reshape(-1)
        self.policy = best_order_actions_df.values.reshape(-1)

        return to_return

    def get_policy(self, V) -> pd.DataFrame:
        """Return the best policy, based on the input values V,
        as a dataframe"""
        # Find that a smaller batch size required for this part
        policy_batch_size = self.batch_size // 2

        # This is slightly round-about way of constructing the table
        # but in practice seemed to help avoid GPU OOM error

        (self.padded_batched_states, self.n_pad,) = self._pad_and_batch_states_for_pmap(
            self.states, policy_batch_size, self.n_devices
        )
        best_action_idxs_padded = self._extract_policy_scan_state_batches_pmap(
            (self.actions, self.possible_random_outcomes, V),
            self.padded_batched_states,
        )
        best_action_idxs = self._unpad(best_action_idxs_padded.reshape(-1), self.n_pad)
        best_order_actions = jnp.take(self.actions, best_action_idxs, axis=0)
        best_order_actions_df = pd.DataFrame(
            np.array(best_order_actions),
            index=tuple(self.state_tuples),
            columns=self.action_labels,
        )
        return best_order_actions_df

    def _setup(self) -> None:
        """Run setup to create arrays of states, actions and random outcomes;
        pmap, vmap and jit methods where required, and load checkpoint if provided"""

        # Manual updates to any parameters will not be reflected in output unless
        # set-up is rerun

        log.info("Starting setup")

        log.info(f"Devices: {jax.devices()}")

        # Vmap and/or JIT methods
        self._deterministic_transition_vmap_random_outcomes = jax.vmap(
            self.problem.deterministic_transition, in_axes=[None, None, 0]
        )
        self._get_value_next_state_vmap_next_states = jax.jit(
            jax.vmap(self._get_value_next_state, in_axes=[0, None])
        )
        self._calculate_updated_state_action_value_vmap_actions = jax.vmap(
            self._calculate_updated_state_action_value, in_axes=[None, 0, None, None]
        )

        self._calculate_updated_value_vmap_states = jax.vmap(
            self._calculate_updated_value, in_axes=[0, None, None, None]
        )
        self._calculate_updated_value_state_batch_jit = jax.jit(
            self._calculate_updated_value_state_batch
        )
        self.calculate_updated_value_scan_state_batches_pmap = jax.pmap(
            self._calculate_updated_value_scan_state_batches,
            in_axes=((None, None, None), 0),
        )

        self._extract_policy_vmap_states = jax.vmap(
            self._extract_policy_one_state, in_axes=[0, None, None, None]
        )
        self._extract_policy_state_batch_jit = jax.jit(self._extract_policy_state_batch)
        self._extract_policy_scan_state_batches_pmap = jax.pmap(
            self._extract_policy_scan_state_batches, in_axes=((None, None, None), 0)
        )

        # Hook for custom setup in subclasses
        self._setup_before_states_actions_random_outcomes_created()

        # Get the states as tuples initially so they can be used to get state_to_idx_mapping
        # before being converted to a jax.numpy array
        self.states = self.problem.get_states()
        self.state_tuples = tuple(np.array(self.states))

        self.n_devices = len(jax.devices())
        self.batch_size = min(
            self.max_batch_size, math.ceil(len(self.states) / self.n_devices)
        )

        # Reshape states into shape (N_devices x N_batches x max_batch_size x state_size)
        self.padded_batched_states, self.n_pad = self._pad_and_batch_states_for_pmap(
            self.states, self.batch_size, self.n_devices
        )

        # Get the possible actions
        self.actions = self.problem.get_actions()
        self.action_labels = self.problem.get_action_labels()

        # Generate the possible random outcomes
        self.possible_random_outcomes = self.problem.get_random_outcomes()


        # Hook for custom setup in subclasses
        self._setup_after_states_actions_random_outcomes_created()

        if not self.resume_from_checkpoint:
            # Initialise the value function
            self.V_old = self.problem.initial_value()
            self.iteration = 1  # start at iteration 1
        else:
            # Allow basic loading of checkpoint for resumption
            log.info(
                f"Loading initial values from checkpoint file: {self.resume_from_checkpoint}"
            )
            loaded_cp_iteration = int(
                re.search("values_(.*).csv", self.resume_from_checkpoint).group(1)
            )
            log.info(
                f"Checkpoint was iteration {loaded_cp_iteration}, so start at iteration {loaded_cp_iteration+1}"
            )
            values_df_loaded = pd.read_csv(
                Path(self.resume_from_checkpoint), index_col=0
            )
            self.V_old = jnp.array(values_df_loaded.iloc[:, 0])
            log.info("Checkpoint loaded")

            self.iteration = (
                loaded_cp_iteration + 1
            )  # first iteration will be after last one in checkpoint

        # Use this to store elements to be reported in res tables
        # for easy collation
        self.output_info = {}
        self.output_info["set_sizes"] = {}
        self.output_info["set_sizes"]["N_states"] = len(self.states)
        self.output_info["set_sizes"]["N_actions"] = len(self.actions)
        self.output_info["set_sizes"]["N_random_outcomes"] = len(
            self.possible_random_outcomes
        )

        # Log some basic information about the problem
        log.info("Setup complete")
        log.warning(
            "Changes to properties of the class after setup will not necessarily be reflected in the output and may lead to errors. To run an experiment with different settings, create a new value iteration runner"
        )
        log.info(f"Output file directory: {self.output_directory}")
        log.info(f"N states = {self.output_info['set_sizes']['N_states']}")
        log.info(f"N actions = {self.output_info['set_sizes']['N_actions']}")
        log.info(
            f"N random outcomes = {self.output_info['set_sizes']['N_random_outcomes']}"
        )

    def _pad_and_batch_states_for_pmap(
        self, states: chex.Array, batch_size: int, n_devices: int
    ) -> Tuple[chex.Array, int]:
        """Pad states and reshape to (N_devices x N_batches x max_batch_size x state_size) to support
        pmap over devices, and using jax.lax.scan to loop over batches of states."""
        n_pad = (n_devices * batch_size) - (len(states) % (n_devices * batch_size))
        padded_states = jnp.vstack(
            [states, jnp.zeros((n_pad, states.shape[1]), dtype=jnp.int32)]
        )
        padded_batched_states = padded_states.reshape(
            n_devices, -1, batch_size, states.shape[1]
        )
        return padded_batched_states, n_pad

    def _unpad(self, padded_array: chex.Array, n_pad: int) -> chex.Array:
        """Remove padding from array"""
        return padded_array[:-n_pad]

    def _get_value_next_state(self, next_state: chex.Array, V_old: chex.Array) -> float:
        """Lookup the value of the next state in the value function from the previous iteration."""
        return V_old[self.problem.get_state_index(tuple(next_state))]
    def _calculate_updated_state_action_value(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        possible_random_outcomes: chex.Array,
        V_old: chex.Array,
    ) -> float:
        """Update the state-action value for a given state, action pair"""
        (
            next_states,
            single_step_rewards,
        ) = self._deterministic_transition_vmap_random_outcomes(
            state,
            action,
            possible_random_outcomes,
        )
        next_state_values = self._get_value_next_state_vmap_next_states(
            next_states, V_old
        )
        probs = self.problem.get_outcome_probabilities(state, action, possible_random_outcomes)
        new_state_action_value = (
            single_step_rewards + self.gamma * next_state_values
        ).dot(probs)
        return new_state_action_value

    def _calculate_updated_value(
        self,
        state: chex.Array,
        actions: Union[int, chex.Array],
        possible_random_outcomes: chex.Array,
        V_old: chex.Array,
    ) -> float:
        """Update the value for a given state, by taking the max of the updated state-action
        values over all actions"""
        return jnp.max(
            self._calculate_updated_state_action_value_vmap_actions(
                state, actions, possible_random_outcomes, V_old
            )
        )

    def _calculate_updated_value_state_batch(
        self, carry, batch_of_states: chex.Array
    ) -> Tuple[Tuple[Union[int, chex.Array], chex.Array, chex.Array], chex.Array]:
        """Calculate the updated value for a batch of states"""
        V = self._calculate_updated_value_vmap_states(batch_of_states, *carry)
        return carry, V

    def _calculate_updated_value_scan_state_batches(
        self,
        carry: Tuple[Union[int, chex.Array], chex.Array, chex.Array],
        padded_batched_states: chex.Array,
    ) -> chex.Array:
        """Calculate the updated value for multiple batches of states, using jax.lax.scan to loop over batches of states."""
        carry, V_padded = jax.lax.scan(
            self._calculate_updated_value_state_batch_jit,
            carry,
            padded_batched_states,
        )
        return V_padded

    def _extract_policy_one_state(
        self,
        state: chex.Array,
        actions: Union[int, chex.Array],
        possible_random_outcomes: chex.Array,
        V: chex.Array,
    ) -> int:
        """Extract the best action for a single state, by taking the argmax of the updated state-action values over all actions"""
        best_action_idx = jnp.argmax(
            self._calculate_updated_state_action_value_vmap_actions(
                state, actions, possible_random_outcomes, V
            )
        )
        return best_action_idx

    def _extract_policy_state_batch(
        self,
        carry: Tuple[Union[int, chex.Array], chex.Array, chex.Array],
        batch_of_states: chex.Array,
    ) -> chex.Array:
        """Extract the best action for a batch of states"""
        best_action_idxs = self._extract_policy_vmap_states(batch_of_states, *carry)
        return carry, best_action_idxs

    def _extract_policy_scan_state_batches(
        self,
        carry: Tuple[Union[int, chex.Array], chex.Array, chex.Array],
        padded_batched_states: chex.Array,
    ) -> chex.Array:
        """Extract the best action for multiple batches of states, using jax.lax.scan to loop over batches of states."""
        carry, best_action_idxs_padded = jax.lax.scan(
            self._extract_policy_state_batch_jit,
            carry,
            padded_batched_states,
        )
        return best_action_idxs_padded
    
    def check_converged(
        self, iteration: int, min_iter: int, V: chex.Array, V_old: chex.Array
    ) -> bool:
        """Standard convergence check to determine whether to stop value iteration. Stops when there is
        approximately (based on epsilon) no change between estimates of the value function at successive iterations"""
        # Here we use a discount factor, so
        # We want biggest change to a value to be less than epsilon
        # This is a difference conv check to the others
        max_delta = jnp.max(jnp.abs(V - V_old))
        if max_delta < self.epsilon:
            if iteration >= min_iter:
                log.info(f"Converged on iteration {iteration}")
                return True
            else:
                log.info(
                    f"Max delta below epsilon on iteration {iteration}, but min iterations not reached"
                )
                return False
        else:
            log.info(f"Iteration {iteration}, max delta: {max_delta}")
            return False