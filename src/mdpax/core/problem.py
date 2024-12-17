"""Base class for defining MDP problems in a structured way."""

from abc import ABC, abstractmethod
from typing import Any, Sequence
import jax.numpy as jnp
from jax import vmap
import jax

class Problem(ABC):
    """Abstract base class for MDP problems."""
    
    @abstractmethod
    def get_states(self) -> jnp.ndarray:
        """Return a sequence of all possible states, shape (n_states,state_size)"""
        pass
    
    @abstractmethod
    def get_state_names(self) -> dict[str, int]:
        """Return a dictionary of that maps from named state components to indices in an individual state."""
        pass

    @abstractmethod
    def get_state_index(self, state: tuple) -> int:
        """Return the index of a state in the state array."""
        pass
    
    @abstractmethod
    def get_actions(self) -> jnp.ndarray:
        """Return an array of all possible actions."""
        pass

    @abstractmethod
    def get_action_labels(self) -> list[str]:
        """Return a list of descriptive names for each action dimension."""
        pass
    
    @abstractmethod
    def get_random_outcomes(self) -> jnp.ndarray:
        """Return an array of all possible random outcomes."""
        pass

    @abstractmethod
    def get_outcome_index(self, outcome: Any) -> jnp.ndarray:
        """Return the index of an outcome in the outcome array."""
        pass
    
    @abstractmethod
    def deterministic_transition(
        self, 
        state: Any, 
        action: jnp.ndarray, 
        outcome: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute next state and reward for a transition."""
        pass
    
    @abstractmethod
    def get_outcome_probabilities(
        self, 
        state: Any, 
        action: jnp.ndarray,
        possible_random_outcomes: jnp.ndarray
    ) -> jnp.ndarray:
        """Return probabilities of the possible random outcomes for a state-action pair.
        
        Returns
        -------
        probs : jnp.ndarray
            Array of probabilities for each outcome, shape [O]
        """
        pass
    
    @abstractmethod
    def initial_value(self) -> jnp.ndarray:
        """Return initial value estimates for all states."""
        pass
    
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
        states = self.get_states()  # [S, state_size]
        actions = self.get_actions()
        outcomes = self.get_random_outcomes()
        
        S = len(states)
        A = len(actions)
        O = len(outcomes)
        
        # Vectorize transition over all dimensions [S, A, O]
        v_transition = vmap(
            vmap(
                vmap(
                    self.deterministic_transition,
                    in_axes=(None, None, 0)  # Outcomes
                ),
                in_axes=(None, 0, None)  # Actions
            ),
            in_axes=(0, None, None)  # States
        )
        
        # Vectorize probability over states and actions [S, A]
        v_probability = vmap(
            vmap(
                lambda s, a: self.get_outcome_probabilities(s, a, outcomes),
                in_axes=(None, 0)  # Actions
            ),
            in_axes=(0, None)  # States
        )
        
        # Get all transitions and probabilities at once
        next_states_rewards = v_transition(states, actions, outcomes)  # [S, A, O]
        next_states, rewards = next_states_rewards  # Unpack tuple
        probs = v_probability(states, actions)  # [S, A, O]
        
        # Convert all next states to indices
        ns_indices = vmap(
            vmap(
                vmap(
                    self.get_state_index,
                    in_axes=0  # Outcomes
                ),
                in_axes=0  # Actions
            ),
            in_axes=0  # States
        )(next_states)  # [S, A, O]
        
        # Initialize matrices
        P = jnp.zeros((A, S, S))
        R = jnp.zeros((S, A))
        
        # Compute expected rewards - sum over outcomes
        # rewards shape: [S, A, O], probs shape: [S, A, O]
        R = jnp.sum(probs * rewards.squeeze(), axis=-1)  # [S, A]
        
        # For each action a, state s, outcome o:
        # Add prob[s,a,o] to P[a,s,ns_indices[s,a,o]]
        for o in range(O):
            # Get indices for this outcome
            ns_idx = ns_indices[:, :, o]  # [S, A]
            p = probs[:, :, o]  # [S, A]
            
            # For each action
            for a in range(A):
                P = P.at[
                    a,                     # Current action
                    jnp.arange(S),         # All source states
                    ns_idx[:, a]           # Next states for this action
                ].add(p[:, a])             # Probabilities for this action
        
        return P, R
