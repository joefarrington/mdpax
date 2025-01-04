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
    To implement a custom problem, subclass this class and implement the following:

    Required Implementations:
        - name (property): Unique identifier for this problem type
        - state_to_index: Convert state vectors to indices
        - _construct_state_space: Define the full state space
        - _construct_action_space: Define the full action space
        - _construct_random_event_space: Define the space of random events
        - random_event_probability: Define transition probabilities
        - transition: Define state transitions and rewards

    Optional Implementations:
        - initial_value: Custom initialization of value function (default: 0.0)
        - _setup_before_space_construction: Custom setup before space construction
        - _setup_after_space_construction: Custom setup after space construction

    Attributes:
        state_space: Array of shape [n_states, state_dim] containing all possible states
        action_space: Array of shape [n_actions, action_dim] containing all possible actions
        random_event_space: Array of shape [n_events, event_dim] containing all possible random events
        n_states: Number of states in the problem
        n_actions: Number of actions in the problem
        n_random_events: Number of random events in the problem
        name: A unique identifier for this problem type

    Shape Requirements:
        - Single state: [state_dim]
        - Single action: [action_dim]
        - Single random event: [event_dim]
        - State space: [n_states, state_dim]
        - Action space: [n_actions, action_dim]
        - Random event space: [n_events, event_dim]

    Note:
        All array operations should be implemented using JAX for compatibility
        with JIT compilation and vmap/pmap.

    See Also:
        For an interactive tutorial on how to implement a custom problem, see
        https://mdpax.readthedocs.io/en/latest/create_custom_problem.html
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
        """Build array of all possible states.

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
            Index of the state in state_space

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
            - Probabilities must sum to 1 over all possible random events
              for each state-action pair
            - This method should be implemented to work efficiently with JAX
              vectorization over batches of states/actions and be compatible
              with JIT compilation
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
            Tuple containing the next state vector [state_dim] and
            the immediate reward

        Note:
            This method should be implemented to work efficiently with JAX
            vectorization over batches of states/actions and be compatible
            with JIT compilation
        """
        pass

    def initial_value(self, state: StateVector) -> float:
        """Return initial value estimate for a given state.

        This method defines how to initialize the value function for a single state.
        The solver will handle vectorization over all states efficiently.

        By default, this method returns 0.0 for all states.

        Args:
            state: State vector [state_dim]

        Returns:
            Initial value estimate for the given state
        """
        return 0.0

    def build_transition_and_reward_matrices(
        self, normalization_tolerance: float = 1e-4
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

        Args:
            normalization_tolerance: If probabilities sum to within this tolerance of 1,
                adjust the largest probability to make them sum exactly to 1.
                Set to 0 to disable this behavior.

        Returns:
            Tuple containing the transition probability matrix
            [n_actions, n_states, n_states] and the
            expected reward matrix [n_states, n_actions]

        Note:
            This method is primarily for testing and comparison purposes.
            It explicitly constructs the full transition matrices which is
            impractical for large state spaces and will result in a memory error.
            The main solver implementations use the transition() method directly
            instead.
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
        # next_states shape: [S, A, E, state_dim]
        # rewards shape: [S, A, E]
        # probs shape: [S, A, E]
        next_states_rewards = v_transition(states, actions, random_events)
        next_states, rewards = next_states_rewards
        probs = v_probability(states, actions, random_events)

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
        # Ensure arrays have shape [S, A, E] even when E=1
        # probs: [S, A, E], rewards: [S, A, E]
        probs = probs.reshape(S, A, E)  # Ensure 3D shape
        rewards = rewards.reshape(S, A, E)  # Ensure 3D shape
        R = jnp.sum(probs * rewards, axis=-1)  # [S, A]

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

        # Check if probabilities sum close enough to 1 before normalizing
        row_sums = jnp.sum(P, axis=-1)  # [A, S]
        max_deviation = jnp.max(jnp.abs(row_sums - 1.0))
        if max_deviation > normalization_tolerance:
            # Find the worst offending state-action pair
            action, state = jnp.unravel_index(
                jnp.argmax(jnp.abs(row_sums - 1.0)), row_sums.shape
            )
            raise ValueError(
                f"Transition probabilities for state {state}, action {action} sum to "
                f"{row_sums[action, state]:.6f}, which deviates from 1.0 by more than "
                f"the tolerance of {normalization_tolerance}"
            )

        # Normalize transition probabilities by dividing each row by its sum
        # Only reaches here if all probabilities are within tolerance
        row_sums = row_sums.reshape(A, S, 1)  # Add dimension for broadcasting
        P = P / jnp.where(row_sums > 0, row_sums, 1.0)  # Avoid division by zero

        # Verify shapes
        chex.assert_shape(P, (A, S, S))
        chex.assert_shape(R, (S, A))

        return P, R
