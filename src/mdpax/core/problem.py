"""Base class for defining MDP problems in a structured way."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import vmap


class Problem(ABC):
    """Abstract base class for MDP problems.

    This class defines the interface for Markov Decision Process (MDP) problems.
    Each problem must define its state space, action space, and random events,
    along with transition dynamics and rewards.
    """

    def __init__(self):
        """Initialize problem with empty spaces - constructed when first accessed."""
        # State space and lookups
        self._state_space = None
        self._state_dimension_sizes = None

        # Action space and lookups
        self._action_space = None

        # Random event space and lookups
        self._random_event_space = None

    @abstractmethod
    def _construct_state_space(self) -> jnp.ndarray:
        """Construct and return the state space [n_states, state_dim]."""
        pass

    @abstractmethod
    def _construct_state_dimension_sizes(self) -> tuple[int, ...]:
        """Return maximum size for each state dimension."""
        pass

    @abstractmethod
    def _construct_action_space(self) -> jnp.ndarray:
        """Construct and return the action space [n_actions, action_dim]."""
        pass

    @abstractmethod
    def _construct_random_event_space(self) -> jnp.ndarray:
        """Construct and return the random event space [n_events, event_dim]."""
        pass

    @property
    def state_dimension_sizes(self) -> tuple[int, ...]:
        """Maximum size for each state dimension."""
        if self._state_dimension_sizes is None:
            self._state_dimension_sizes = self._construct_state_dimension_sizes()
        return self._state_dimension_sizes

    @property
    def state_space(self) -> jnp.ndarray:
        """Array of all possible states [n_states, state_dim]."""
        if self._state_space is None:
            self._state_space = self._construct_state_space()
        return self._state_space

    @property
    def action_space(self) -> jnp.ndarray:
        """Array of all possible actions [n_actions, action_dim]."""
        if self._action_space is None:
            self._action_space = self._construct_action_space()
        return self._action_space

    @property
    def random_event_space(self) -> jnp.ndarray:
        """Array of all possible random events [n_events, event_dim]."""
        if self._random_event_space is None:
            self._random_event_space = self._construct_random_event_space()
        return self._random_event_space

    @property
    def n_states(self) -> int:
        """Number of states in the problem."""
        return len(self.state_space)

    @property
    def n_actions(self) -> int:
        """Number of actions in the problem."""
        return len(self.action_space)

    @property
    def n_random_events(self) -> int:
        """Number of random events in the problem."""
        return len(self.random_event_space)

    @abstractmethod
    def action_components(self) -> Dict[str, int]:
        """Return list of action component names."""
        pass

    @abstractmethod
    def transition(
        self, state: jnp.ndarray, action: jnp.ndarray, random_event: jnp.ndarray
    ) -> Tuple[jnp.ndarray, float]:
        """Compute next state and reward for a transition."""
        pass

    @abstractmethod
    def random_event_probability(
        self, state: jnp.ndarray, action: jnp.ndarray, random_event: jnp.ndarray
    ) -> float:
        """Return probability of a single random event given state-action pair.

        Parameters
        ----------
        state : jnp.ndarray
            Current state [state_dim]
        action : jnp.ndarray
            Selected action [action_dim]
        random_event : jnp.ndarray
            Single random event [event_dim]

        Returns
        -------
        float
            Probability of this random event occurring
        """
        pass

    @abstractmethod
    def initial_values(self) -> jnp.ndarray:
        """Return initial state values for value-based methods."""
        pass

    def state_to_index(self, state: jnp.ndarray) -> int:
        """Convert state vector to index."""
        return jnp.ravel_multi_index(
            tuple(state), self.state_dimension_sizes, mode="wrap"
        )

    def build_matrices(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Build transition and reward matrices.

        Returns
        -------
        P : jnp.ndarray
            Transition matrices, shape [A, S, S].
            P[a,s,s'] is probability of going from s to s' under action a.
        R : jnp.ndarray
            Reward matrix, shape [S, A].
            R[s,a] is reward for taking action a in state s.
        """
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

        # Vectorize probability over all dimensions [S, A, E]
        v_probability = vmap(
            vmap(
                vmap(
                    self.random_event_probability,
                    in_axes=(None, None, 0),  # Random events
                ),
                in_axes=(None, 0, None),  # Actions
            ),
            in_axes=(0, None, None),  # States
        )

        # Get all transitions and probabilities at once
        next_states_rewards = v_transition(states, actions, random_events)  # [S, A, E]
        next_states, rewards = next_states_rewards  # Unpack tuple
        probs = v_probability(states, actions, random_events)  # [S, A, E]

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
