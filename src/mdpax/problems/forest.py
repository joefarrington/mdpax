"""Forest management MDP problem."""

from dataclasses import dataclass

import jax.numpy as jnp

from mdpax.core.problem import Problem, ProblemConfig
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
class ForestConfig(ProblemConfig):
    """Configuration for the Forest Management problem.


    Attributes:
        S: Number of states (tree ages from 0 to S-1)
        r1: Reward for waiting when forest in oldest state
        r2: Reward for cutting when forest in oldest state
        p: Base probability of fire
    """

    _target_: str = "mdpax.problems.forest.Forest"
    S: int = 3
    r1: float = 4.0
    r2: float = 2.0
    p: float = 0.1


class Forest(Problem):
    """Forest management MDP problem.

    The forest management problem involves deciding whether to cut down trees
    for immediate reward or wait for them to grow larger. There is a risk of
    fire destroying the forest during each time step.

    Adapted from the example problem in Python MDP Toolbox
    https://github.com/sawcordwell/pymdptoolbox/blob/master/src/mdptoolbox/example.py

    State Space (state_dim = 1):
        Vector containing:
        - Tree age: 1 element in range [0, S-1] (newly planted to mature forest)

    Action Space (action_dim = 1):
        Vector containing:
        - Decision: 1 element in range {0=wait, 1=cut}

    Random Events (event_dim = 1):
        Vector containing:
        - Fire occurrence: 1 element in range {0=no_fire, 1=fire}

    Dynamics:
        1. Choose to cut or wait
        2. If cut, receive reward r2 if in oldest state and 1 otherwise
            and reset to age 0
        3. If wait:
            - Check for fire (probability p if waiting, 0 if cut)
            - If fire, reset to age 0 with no reward
            - If no fire, age increases by 1 (up to S-1) and receive
              r1 reward if in oldest state

    Args:
        S: Number of states (tree ages from 0 to S-1)
        r1: Reward for waiting when forest in oldest state
        r2: Reward for cutting when forest in oldest state
        p: Base probability of fire
    """

    def __init__(self, S: int = 3, r1: float = 4.0, r2: float = 2.0, p: float = 0.1):
        assert S > 0, "Number of states must be positive"
        assert 0 <= p <= 1, "Probability must be between 0 and 1"

        self.S = S
        self.r1 = r1
        self.r2 = r2
        self.p = p
        self.probability_matrix = jnp.array([[1 - self.p, self.p], [1, 0]])
        super().__init__()

    @property
    def name(self) -> str:
        """Name of the problem."""
        return "forest"

    def _construct_state_space(self) -> StateSpace:
        """Build array of all possible states.

        Returns:
            Array of shape [n_states, state_dim] containing all possible tree ages
        """
        return jnp.arange(self.S, dtype=jnp.int32).reshape(-1, 1)

    def state_to_index(self, state: StateVector) -> int:
        """Convert state vector to index.

        Args:
            state: State vector to convert [state_dim]

        Returns:
            Integer index of the state in state_space
        """
        return state[0]

    def _construct_action_space(self) -> ActionSpace:
        """Build array of all possible actions.

        Returns:
            Array of shape [n_actions, action_dim] containing actions [wait=0, cut=1]
        """
        return jnp.array([[0], [1]], dtype=jnp.int32)

    def _construct_random_event_space(self) -> RandomEventSpace:
        """Build array of all possible random events.

        Returns:
            Array of shape [n_events, event_dim] containing events [no_fire=0, fire=1]
        """
        return jnp.array([[0], [1]], dtype=jnp.int32)

    def random_event_probability(
        self, state: StateVector, action: ActionVector, random_event: RandomEventVector
    ) -> float:
        """Compute probability of random event given state-action pair.

        When waiting:
            - No fire probability is 1 - p
            - Fire probability is p
        When cutting:
            - No fire probability is 1
            - Fire probability is 0

        Args:
            state: Current tree age [state_dim]
            action: Cut (1) or wait (0) [action_dim]
            random_event: Fire (1) or no fire (0) [event_dim]

        Returns:
            Probability of the random event occurring
        """
        return self.probability_matrix[action[0], random_event[0]]

    def transition(
        self, state: StateVector, action: ActionVector, random_event: RandomEventVector
    ) -> tuple[StateVector, Reward]:
        """Compute next state and reward for forest transition.

        The reward is 0 for waiting except in the final state, where it is r1.
        The reward is 1 for cutting except in the final state, where it is r2.

        Args:
            state: Current tree age [state_dim]
            action: Cut (1) or wait (0) [action_dim]
            random_event: Fire (1) or no fire (0) [event_dim]

        Returns:
            Tuple of (next_state, reward) where:
                - next_state is the next tree age [state_dim]
                - reward is the immediate reward for this transition
        """
        is_cut = action[0] == 1
        is_fire = random_event[0] == 1

        # Compute reward - only get reward when cutting
        reward = jnp.where(
            is_cut,
            # Cut reward depends on tree age
            jnp.where(
                state[0] == self.S - 1, self.r2, jnp.where(state[0] == 0, 0.0, 1.0)
            ),
            # No reward for waiting except in final state
            jnp.where(state[0] == self.S - 1, self.r1, 0.0),
        )

        # Compute next state
        next_state = jnp.array(
            [
                jnp.where(
                    is_cut | is_fire,
                    # Reset to age 0 if cut or fire
                    0,
                    # Otherwise increment age up to S-1
                    jnp.minimum(state[0] + 1, self.S - 1),
                )
            ]
        ).astype(jnp.int32)

        return next_state, reward

    def get_problem_config(self) -> ForestConfig:
        """Get problem configuration for reconstruction.

        Returns:
            Configuration containing all parameters needed to reconstruct
            this problem instance
        """
        return ForestConfig(
            S=self.S,
            r1=float(self.r1),
            r2=float(self.r2),
            p=float(self.p),
        )
