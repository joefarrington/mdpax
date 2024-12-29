"""Base class for defining MDP problems in a structured way."""

from abc import ABC, abstractmethod

import chex
import jax.numpy as jnp
from hydra.conf import MISSING, dataclass
from jax import vmap
from jaxtyping import Array, Float

from mdpax.utils.types import (
    ActionSpace,
    ActionVector,
    RandomEventSpace,
    RandomEventVector,
    Reward,
    StateSpace,
    StateVector,
)


@dataclass
class ProblemConfig:
    """Base configuration for all MDP problems.

    This serves as the base configuration class that all specific problem
    configurations should inherit from. It enforces that all problems must
    specify their target class.

    Attributes:
        _target_: Full path to the problem class for Hydra instantiation
    """

    _target_: str = MISSING


class Problem(ABC):
    """Abstract base class for MDP problems.

    This class defines the interface for Markov Decision Process (MDP) problems.
    Each problem must methods for constructing its state space, action space,
    and random event space, along with transition dynamics and rewards.

    The interface is designed for efficient computation with JAX, supporting
    vectorization and parallel processing. States, actions, and random events
    are represented as vectors, and operations are designed to work with batches.

    Shape Requirements:
        All inputs/outputs maintain consistent dimensionality:
        - Single state: [state_dim]
        - Single action: [action_dim]
        - Single random event: [event_dim]
        - State space: [n_states, state_dim]
        - Action space: [n_actions, action_dim]
        - Random event space: [n_events, event_dim]

    Note:
        All array operations should be implemented using JAX for compatibility
        with JIT compilation and vmap/pmap.
    """

    def __init__(self):
        """Initialize problem with all spaces and lookups constructed immediately."""
        self._setup_before_space_construction()
        self._state_space = self._ensure_2d_space(self._construct_state_space())
        self._action_space = self._ensure_2d_space(self._construct_action_space())
        self._random_event_space = self._ensure_2d_space(
            self._construct_random_event_space()
        )
        self._setup_after_space_construction()

    def _ensure_2d_space(self, x: Float[Array, "*dim"]) -> Float[Array, "n dim"]:
        """Ensure space array is 2D by adding feature dimension if needed.

        Args:
            x: Input array that may be 1D [n] or 2D [n, dim]

        Returns:
            Array reshaped to [n, 1] if needed
        """
        return x.reshape(-1, 1) if x.ndim == 1 else x

    @property
    @abstractmethod
    def name(self) -> str:
        """A unique identifier for this problem type"""
        pass

    def _setup_before_space_construction(self) -> None:
        """Setup operations needed before constructing spaces."""
        pass

    def _setup_after_space_construction(self) -> None:
        """Setup operations run after constructing spaces."""
        pass

    # State Space Methods
    @property
    def state_space(self) -> StateSpace:
        """Array of shape [n_states, state_dim] containing all possible states"""
        return self._state_space

    @property
    def n_states(self) -> int:
        """Number of states in the problem."""
        return len(self.state_space)

    @abstractmethod
    def _construct_state_space(self) -> StateSpace:
        """Build an array of all possible states.

        Returns:
            Array of shape [n_states, state_dim] containing all possible states
        """
        pass

    @abstractmethod
    def state_to_index(self, state: StateVector) -> int:
        """Convert state vector to index.

        Args:
            state: Vector representation of a state [state_dim]

        Returns:
            Integer index of the state in state_space

        Note:
            This mapping must be consistent with the ordering in state_space
        """
        pass

    # Action Space Methods
    @property
    def action_space(self) -> ActionSpace:
        """Array of shape [n_actions, action_dim] containing all possible actions"""
        return self._action_space

    @property
    def n_actions(self) -> int:
        """Number of actions in the problem."""
        return len(self.action_space)

    @abstractmethod
    def _construct_action_space(self) -> ActionSpace:
        """Build an array of all possible actions.

        Returns:
            Array of shape [n_actions, action_dim] containing all possible actions
        """
        pass

    # Random event Methods
    @property
    def random_event_space(self) -> RandomEventSpace:
        """Array of shape [n_events, event_dim] containing all possible random events"""
        return self._random_event_space

    @property
    def n_random_events(self) -> int:
        """Number of random events in the problem."""
        return len(self.random_event_space)

    @abstractmethod
    def _construct_random_event_space(self) -> RandomEventSpace:
        """Build an array of all possible random events.

        Returns:
            Array of shape [n_events, event_dim] containing all possible random events
        """
        pass

    @abstractmethod
    def random_event_probability(
        self, state: StateVector, action: ActionVector, random_event: RandomEventVector
    ) -> float:
        """Calculate probability of random event given state-action pair.

        Args:
            state: Current state vector [state_dim]
            action: Action vector [action_dim]
            random_event: Random event vector [event_dim]

        Returns:
            Probability of the random event occurring

        Note:
            Probabilities must sum to 1 over all possible random events
            for each state-action pair
        """
        pass

    # Core MDP Methods
    @abstractmethod
    def transition(
        self, state: StateVector, action: ActionVector, random_event: RandomEventVector
    ) -> tuple[StateVector, Reward]:
        """Compute next state and reward for a transition.

        Args:
            state: Current state vector [state_dim]
            action: Action vector [action_dim]
            random_event: Random event vector [event_dim]

        Returns:
            Tuple of (next_state, reward) where:
                - next_state is the resulting state vector [state_dim]
                - reward is the immediate reward for this transition

        Note:
            This method should be implemented to work efficiently with JAX
            vectorization over batches of states/actions
        """
        pass

    def initial_value(self, state: StateVector) -> float:
        """Return initial value estimate for a given state.

        This method defines how to initialize the value function for a single state.
        The solver will handle vectorization over all states efficiently.

        Args:
            state: State vector [state_dim]

        Returns:
            Initial value estimate for the given state
        """
        return 0.0

    # Support methods
    @abstractmethod
    def get_problem_config(self) -> ProblemConfig:
        """Get problem configuration for reconstruction.

        Returns:
            Configuration containing all parameters needed to reconstruct
            this problem instance

        Note:
            This is used during checkpoint restoration to recreate the problem
        """
        pass

    def build_matrices(
        self,
    ) -> tuple[
        Float[Array, "n_actions n_states n_states"], Float[Array, "n_states n_actions"]
    ]:
        """Build transition and reward matrices for the MDP.

        This method constructs the full transition probability and reward matrices
        for comparison with other solvers (e.g., mdptoolbox) on small problems.
        Not recommended for large state/action spaces.

        The transition probability matrix P has shape [n_actions, n_states, n_states] where:
        - P[a,s,s'] is the probability of transitioning from state s to s' under action a

        The reward matrix R has shape [n_states, n_actions] where:
        - R[s,a] is the expected immediate reward for taking action a in state s

        Returns:
            tuple containing:
                - P: Transition probability matrix [n_actions, n_states, n_states]
                - R: Expected reward matrix [n_states, n_actions]

        Note:
            This method is primarily for testing and comparison purposes.
            It explicitly constructs the full transition matrices which is
            impractical for large state spaces. The main solver implementations
            use the transition() method directly instead.
        """
        states: StateSpace = self.state_space  # [S, state_dim]
        actions: ActionSpace = self.action_space  # [A, action_dim]
        random_events: RandomEventSpace = self.random_event_space  # [E, event_dim]

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
        P: Float[Array, "n_actions n_states n_states"] = jnp.zeros((A, S, S))
        R: Float[Array, "n_states n_actions"] = jnp.zeros((S, A))

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

        # Verify shapes
        chex.assert_shape(P, (A, S, S))
        chex.assert_shape(R, (S, A))

        return P, R
