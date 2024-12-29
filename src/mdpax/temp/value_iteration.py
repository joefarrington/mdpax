import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from loguru import logger


class NewValueIterationRunner:
    def __init__(
        self,
        problem,
        max_batch_size: int,
        epsilon: float,
        gamma: float,
        output_directory: Optional[Union[str, Path]] = None,
        checkpoint_frequency: int = 0,
        resume_from_checkpoint: Union[bool, str] = False,
        periodic_convergence_check: bool = True,
    ):
        """Base class for running value iteration

        Args:
            max_batch_size: Maximum number of states to update in parallel using vmap,
              will depend on GPU memory
            epsilon: Convergence criterion for value iteration
            gamma: Discount factor
            output_directory: Directory to save output to, if None, will create a
            new directory
            checkpoint_frequency: Frequency with which to save checkpoints, 0 for
            no checkpoints
            resume_from_checkpoint: If False, start from scratch; if filename,
            resume from checkpoint

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

        if self.checkpoint_frequency > 0:
            self.cp_path = self.output_directory / "checkpoints"
            self.cp_path.mkdir(parents=True, exist_ok=True)

        self.resume_from_checkpoint = resume_from_checkpoint
        self.periodic_convergence_check = periodic_convergence_check

        self._setup()

        if periodic_convergence_check:
            assert (
                self.checkpoint_frequency == 1
            ), "Checkpoint frequency must be 1 to use periodic convergence check"

            # Save an initial checkpoint of V at iteration 0 for use
            # by periodic conv check
            V_0 = self.problem.calculate_initial_values(self.states)
            values_df = pd.DataFrame(
                np.array(V_0), index=self.state_tuples, columns=["V"]
            )
            values_df.to_csv(self.cp_path / "values_0.csv")

    def check_converged(
        self, iteration: int, min_iter: int, V: chex.Array, V_old: chex.Array
    ) -> bool:
        """Convergence check to determine whether to stop value iteration. We allow the
        choice of two convergence checks based on the value of
        self.periodic_convergence_check: the periodic convergence check will generally
        require fewer iterations, but the standard convergence check can be used if
        the values are needed"""
        # By default, we used periodic convergence check
        # because we're interested in the policy rather than the value function
        if self.periodic_convergence_check:
            return self._check_converged_periodic(iteration, min_iter, V, V_old)
        else:
            # But we can also check for convergence of value function itself
            # as in DeMoor case
            return self._check_converged_v(iteration, min_iter, V, V_old)

    def _check_converged_periodic(
        self, iteration: int, min_iter: int, V: chex.Array, V_old: chex.Array
    ) -> bool:
        """Periodic convergence check to determine whether to stop value iteration.
        Stops when the (undiscounted) change in
        a value over a period is the same for every state. The periodicity is 7 -
        the days of the week
        """
        period = len(self.problem.weekdays.keys())
        if iteration < (period):
            logger.info(
                f"Iteration {iteration} complete, but fewer iterations than\
                      periodicity so cannot check for convergence yet"
            )
            return False
        else:
            if self.gamma == 1:
                (
                    min_period_delta,
                    max_period_delta,
                ) = self._calculate_period_deltas_without_discount(V, iteration, period)
            else:
                (
                    min_period_delta,
                    max_period_delta,
                ) = self._calculate_period_deltas_with_discount(
                    V, iteration, period, self.gamma
                )
            delta_diff = max_period_delta - min_period_delta
            if (
                delta_diff
                <= 2
                * self.epsilon
                * jnp.array(
                    [jnp.abs(min_period_delta), jnp.abs(max_period_delta)]
                ).min()
            ):
                if iteration >= min_iter:
                    logger.info(f"Converged on iteration {iteration}")
                    logger.info(f"Max period delta: {max_period_delta}")
                    logger.info(f"Min period delta: {min_period_delta}")
                    return True
                else:
                    logger.info(
                        f"Difference below epsilon on iteration {iteration},\
                              but min iterations not reached"
                    )
                    return False
            else:
                logger.info(f"Iteration {iteration}, period delta diff: {delta_diff}")
                return False

    def _check_converged_v(
        self, iteration: int, min_iter: int, V: chex.Array, V_old: chex.Array
    ) -> bool:
        """Standard convergence check to determine whether to stop value iteration.
        Stops when there is
        approximately (based on epsilon) no change between estimates of the
        value function at successive iterations
        """
        # Here we use a discount factor, so
        # We want biggest change to a value to be less than epsilon
        # This is a difference conv check to the others
        max_delta = jnp.max(jnp.abs(V - V_old))
        if max_delta < self.epsilon:
            if iteration >= min_iter:
                logger.info(f"Converged on iteration {iteration}")
                return True
            else:
                logger.info(
                    f"Max delta below epsilon on iteration {iteration}, but \
                          min iterations not reached"
                )
                return False
        else:
            logger.info(f"Iteration {iteration}, max delta: {max_delta}")
            return False

    ### End of essential methods to implement in subclass ###

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
        max_iter: int = 100,
        min_iter: int = 1,
        extract_policy: bool = True,
    ) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """Run value iteration for a given number of iterations, or until convergence.
          Optionally save checkpoints of the

        Args:
            value function after each iteration, and the final value function and
            policy at the end of the run.
            max_iter: maximum number of iterations to run
            min_iter: minimum number of iterations to run, even if convergence is
            reached before this
            extract_policy: whether to save the final policy as a csv file

        Returns:
            A dictionary containing information to log, the final value function and,
            optionally, the policy

        """
        # If already run more than max_iter, raise an error
        if self.iteration > max_iter:
            raise ValueError(
                f"At least {max_iter} iterations have already been completed"
            )

        # If min_iter greater than max_iter, raise an error
        if min_iter > max_iter:
            raise ValueError("min_iter must be less than or equal to max_iter")

        logger.info(f"Starting value iteration at iteration {self.iteration}")

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
                    values_df = pd.DataFrame(V, index=self.state_tuples, columns=["V"])
                    values_df.to_csv(self.cp_path / f"values_{i}.csv")

                self.V_old = V

            self.iteration += 1

        to_return = {}

        # Put final values into pd.DataFrame to return
        values_df = pd.DataFrame(np.array(V), index=self.state_tuples, columns=["V"])
        to_return["V"] = values_df

        # If extract_policy is True, extract policy with one-step ahead search
        # to return in output
        if extract_policy:
            logger.info("Extracting final policy")
            best_order_actions_df = self.get_policy(V)
            logger.info("Final policy extracted")
            to_return["policy"] = best_order_actions_df

        to_return["output_info"] = self.output_info

        return to_return

    def get_policy(self, V) -> pd.DataFrame:
        """Return the best policy, based on the input values V,
        as a dataframe"""
        # Find that a smaller batch size required for this part
        policy_batch_size = self.batch_size // 2

        # This is slightly round-about way of constructing the table
        # but in practice seemed to help avoid GPU OOM error

        (
            self.padded_batched_states,
            self.n_pad,
        ) = self._pad_and_batch_states_for_pmap(
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
            index=self.state_tuples,
            columns=self.action_labels,
        )
        return best_order_actions_df

    def _setup(self) -> None:
        """Run setup to create arrays of states, actions and random outcomes;
        pmap, vmap and jit methods where required, and load checkpoint if provided"""

        # Manual updates to any parameters will not be reflected in output unless
        # set-up is rerun

        logger.info("Starting setup")

        logger.info(f"Devices: {jax.devices()}")

        # Vmap and/or JIT methods
        self._deterministic_transition_function_vmap_random_outcomes = jax.vmap(
            self.problem.deterministic_transition_function, in_axes=[None, None, 0]
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

        # Get the states as tuples initially so they can be used to get
        # state_to_idx_mapping
        # before being converted to a jax.numpy array
        self.state_tuples = self.problem.generate_states()
        self.problem.state_component_lookup = (
            self.problem._construct_state_component_lookup()
        )
        self.state_to_idx_mapping = self.problem.create_state_to_idx_mapping(
            self.state_tuples
        )
        self.states = jnp.array(np.array(self.state_tuples))

        self.n_devices = len(jax.devices())
        self.batch_size = min(
            self.max_batch_size, math.ceil(len(self.states) / self.n_devices)
        )

        # Reshape states into shape (N_devices x N_batches x max_batch_size
        # x state_size)
        self.padded_batched_states, self.n_pad = self._pad_and_batch_states_for_pmap(
            self.states, self.batch_size, self.n_devices
        )

        # Get the possible actions
        self.actions, self.action_labels = self.problem.generate_actions()

        # Generate the possible random outcomes
        (
            self.possible_random_outcomes,
            self.problem.pro_component_idx_dict,
        ) = self.problem.generate_possible_random_outcomes()
        self.problem.event_component_lookup = (
            self.problem._construct_event_component_lookup()
        )

        self.problem.random_event_space = self.possible_random_outcomes

        # Hook for custom setup in subclasses
        self._setup_after_states_actions_random_outcomes_created()

        if not self.resume_from_checkpoint:
            # Initialise the value function
            self.V_old = self.problem.calculate_initial_values(self.states)
            self.iteration = 1  # start at iteration 1
        else:
            # Allow basic loading of checkpoint for resumption
            logger.info(
                f"Loading initial values from checkpoint file\
                    : {self.resume_from_checkpoint}"
            )
            loaded_cp_iteration = int(
                re.search("values_(.*).csv", self.resume_from_checkpoint).group(1)
            )
            logger.info(
                f"Checkpoint was iteration {loaded_cp_iteration}, so start at\
                      iteration {loaded_cp_iteration+1}"
            )
            values_df_loaded = pd.read_csv(
                Path(self.resume_from_checkpoint), index_col=0
            )
            self.V_old = jnp.array(values_df_loaded.iloc[:, 0])
            logger.info("Checkpoint loaded")

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
        logger.info("Setup complete")
        logger.warning(
            "Changes to properties of the class after setup will not necessarily \
                be reflected in the output and may lead to errors. To run an experiment\
                      with different settings, create a new value iteration runner"
        )
        logger.info(f"Output file directory: {self.output_directory}")
        logger.info(f"N states = {self.output_info['set_sizes']['N_states']}")
        logger.info(f"N actions = {self.output_info['set_sizes']['N_actions']}")
        logger.info(
            f"N random outcomes = {self.output_info['set_sizes']['N_random_outcomes']}"
        )

    def _pad_and_batch_states_for_pmap(
        self, states: chex.Array, batch_size: int, n_devices: int
    ) -> Tuple[chex.Array, int]:
        """Pad states and reshape to (N_devices x N_batches x max_batch_size x
          state_size) to support
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
        """Lookup the value of the next state in the value function from the
        previous iteration."""
        return V_old[self.state_to_idx_mapping[tuple(next_state)]]

    def _calculate_updated_state_action_value(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        possible_random_outcomes: chex.Array,
        V_old: chex.Array,
    ) -> float:
        """Update the state-action value for a given state, action pair"""
        logger.info(
            f"Calculating updated state-action value\
                  for state {state} and action {action}"
        )
        (
            next_states,
            single_step_rewards,
        ) = self._deterministic_transition_function_vmap_random_outcomes(
            state,
            action,
            possible_random_outcomes,
        )
        next_state_values = self._get_value_next_state_vmap_next_states(
            next_states, V_old
        )
        probs = self.problem.get_probabilities(state, action, possible_random_outcomes)
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
        """Update the value for a given state, by taking the max of the
          updated state-action
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
        """Calculate the updated value for multiple batches of states, using
        jax.lax.scan to loop over batches of states."""
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
        """Extract the best action for a single state, by taking the argmax of the
        updated state-action values over all actions"""
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
        """Extract the best action for multiple batches of states, using jax.lax.scan
        o loop over batches of states."""
        carry, best_action_idxs_padded = jax.lax.scan(
            self._extract_policy_state_batch_jit,
            carry,
            padded_batched_states,
        )
        return best_action_idxs_padded

    def _calculate_period_deltas_without_discount(
        self, V: chex.Array, current_iteration: int, period: int
    ) -> Tuple[float, float]:
        """Return the min and max change in the value function over a period if
        there is no discount factor"""
        # If there's no discount factor, just subtract Values one period ago
        # from current value estimate
        fname = self.cp_path / f"values_{current_iteration - period}.csv"
        V_one_period_ago_df = pd.read_csv(fname, index_col=0)
        V_one_period_ago = jnp.array(V_one_period_ago_df.values).reshape(-1)
        max_period_delta = jnp.max(V - V_one_period_ago)
        min_period_delta = jnp.min(V - V_one_period_ago)
        return min_period_delta, max_period_delta

    def _calculate_period_deltas_with_discount(
        self, V: chex.Array, current_iteration: int, period: int, gamma: float
    ) -> Tuple[float, float]:
        """Return the min and max undiscounted change in the value function
        over a period
        if there is a discount factor"""
        # If there is a discount factor, we need to sum the differences between
        # each step in the period and adjust for the discount factor
        values_dict = self._read_multiple_previous_values(current_iteration, period)
        values_dict[current_iteration] = V
        period_deltas = jnp.zeros_like(V)
        for p in range(period):
            period_deltas += (
                values_dict[current_iteration - p]
                - values_dict[current_iteration - p - 1]
            ) / (gamma ** (current_iteration - p - 1))
        min_period_delta = jnp.min(period_deltas)
        max_period_delta = jnp.max(period_deltas)
        return min_period_delta, max_period_delta

    def _read_multiple_previous_values(
        self, current_iteration: int, period: int
    ) -> Dict[int, chex.Array]:
        """Load the value functions from multiple previous iterations fo
        calculate period deltas"""
        values_dict = {}
        for p in range(1, period + 1):
            j = current_iteration - p
            fname = self.cp_path / f"values_{j}.csv"
            values_dict[j] = jnp.array(pd.read_csv(fname)["V"].values)
        return values_dict
