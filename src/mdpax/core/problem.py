"""Base class for defining MDP problems in a structured way."""

import itertools
from abc import ABC, abstractmethod
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax import vmap


class Problem(ABC):
    """Abstract base class for MDP problems.

    This class defines the interface for Markov Decision Process (MDP) problems.
    Each problem must define its state space, action space, and random events,
    along with transition dynamics and rewards.
    """

    def __init__(self):
        """Initialize problem with all spaces and lookups constructed immediately."""
        self._state_bounds = self._construct_state_bounds()
        self._state_dimension_sizes = self._construct_state_dimension_sizes()
        self._state_space = self._construct_state_space()
        self._action_space = self._construct_action_space()
        self._random_event_space = self._construct_random_event_space()

    # State Space Methods
    @property
    def state_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Min and max values for each state dimension."""
        return self._state_bounds

    @property
    def state_space(self) -> jnp.ndarray:
        """Array of all possible states [n_states, state_dim]."""
        return self._state_space

    @property
    def state_dimension_sizes(self) -> tuple[int, ...]:
        """Size of each dimension computed from bounds."""
        return self._state_dimension_sizes

    @property
    def n_states(self) -> int:
        """Number of states in the problem."""
        return len(self.state_space)

    @abstractmethod
    def _construct_state_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (min_values, max_values) for each state dimension."""
        pass

    def _construct_state_space(self) -> jnp.ndarray:
        """Construct state space from bounds."""
        mins, maxs = self.state_bounds
        ranges = [
            np.arange(min_val, max_val + 1) for min_val, max_val in zip(mins, maxs)
        ]
        # More efficient to convert to numpy array first
        states = np.array(list(itertools.product(*ranges)), dtype=jnp.int32)
        return jnp.array(states)

    def _construct_state_dimension_sizes(self) -> tuple[int, ...]:
        """Return maximum size for each state dimension."""
        mins, maxs = self.state_bounds
        return tuple(jnp.array(maxs - mins + 1, dtype=jnp.int32))

    def state_to_index(self, state: jnp.ndarray) -> int:
        """Convert state vector to index."""
        return jnp.ravel_multi_index(
            tuple(state), self.state_dimension_sizes, mode="wrap"
        )

    # Action Space Methods
    @property
    def action_space(self) -> jnp.ndarray:
        """Array of all possible actions [n_actions, action_dim]."""
        return self._action_space

    @property
    def n_actions(self) -> int:
        """Number of actions in the problem."""
        return len(self.action_space)

    @abstractmethod
    def _construct_action_space(self) -> jnp.ndarray:
        """Construct and return the action space [n_actions, action_dim]."""
        pass

    @abstractmethod
    def action_components(self) -> list[str]:
        """Return list of action component names."""
        pass

    # Random Event Methods
    @property
    def random_event_space(self) -> jnp.ndarray:
        """Array of all possible random events [n_events, event_dim]."""
        return self._random_event_space

    @property
    def n_random_events(self) -> int:
        """Number of random events in the problem."""
        return len(self.random_event_space)

    @abstractmethod
    def _construct_random_event_space(self) -> jnp.ndarray:
        """Construct and return the random event space [n_events, event_dim]."""
        pass

    @abstractmethod
    def random_event_probabilities(
        self, state: jnp.ndarray, action: jnp.ndarray
    ) -> float:
        """Return probabilities of each random events given state-action pair."""
        pass

    # Core MDP Methods
    @abstractmethod
    def transition(
        self, state: jnp.ndarray, action: jnp.ndarray, random_event: jnp.ndarray
    ) -> Tuple[jnp.ndarray, float]:
        """Compute next state and reward for a transition."""
        pass

    @abstractmethod
    def initial_values(self) -> jnp.ndarray:
        """Return initial state values for value-based methods."""
        pass

    def build_matrices(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Build transition and reward matrices."""
        states = self.state_space  # [S, state_dim]
        actions = self.action_space
        random_events = self.random_event_space

        S = self.n_states
        A = self.n_actions
        E = self.n_random_events

        # Vectorize transition over all dimensions [S, A, E]
        v_transition = vmap(
            vmap(
                vmap(self.transition, in_axes=(None, None, 0)),  # Random events
                in_axes=(None, 0, None),  # Actions
            ),
            in_axes=(0, None, None),  # States
        )

        # Vectorize probability over states and actions [S, A]
        v_probability = vmap(
            vmap(
                self.random_event_probabilities,
                in_axes=(None, 0),  # Actions
            ),
            in_axes=(0, None),  # States
        )

        # Get all transitions and probabilities at once
        next_states_rewards = v_transition(states, actions, random_events)  # [S, A, E]
        next_states, rewards = next_states_rewards  # Unpack tuple
        probs = v_probability(states, actions)  # [S, A, E]

        # Convert all next states to indices
        ns_indices = vmap(
            vmap(
                vmap(self.state_to_index, in_axes=0),  # Random events
                in_axes=0,  # Actions
            ),
            in_axes=0,  # States
        )(
            next_states
        )  # [S, A, E]

        # Initialize matrices
        P = jnp.zeros((A, S, S))
        R = jnp.zeros((S, A))

        # Compute expected rewards - sum over random events
        R = jnp.sum(probs * rewards.squeeze(), axis=-1)  # [S, A]

        # For each action a, state s, random event e:
        # Add prob[s,a,e] to P[a,s,ns_indices[s,a,e]]
        for e in range(E):
            # Get indices for this random event
            ns_idx = ns_indices[:, :, e]  # [S, A]
            p = probs[:, :, e]  # [S, A]

            # For each action
            for a in range(A):
                P = P.at[
                    a,  # Current action
                    jnp.arange(S),  # All source states
                    ns_idx[:, a],  # Next states for this action
                ].add(
                    p[:, a]
                )  # Probabilities for this action

        return P, R
