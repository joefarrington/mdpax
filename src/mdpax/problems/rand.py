import jax.numpy as jnp
import numpy as np

from mdpax.core.problem import Problem


class RandDense(Problem):
    """Random dense MDP problem.

    Args:
        states: Number of states (> 1)
        actions: Number of actions (> 1)
        mask: Optional array with 0 and 1 (0 indicates zero probability)
              Shape can be (S, S) or (A, S, S). Default: random
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        states: int = 4,
        actions: int = 3,
        mask: np.ndarray = None,
        seed: int = None,
    ):
        assert states > 1, "Number of states must be greater than 1"
        assert actions > 1, "Number of actions must be greater than 1"

        if seed is not None:
            np.random.seed(seed)

        self.states = states
        self.actions = actions
        self.mask = mask

        # Generate P and R matrices
        self.P, self.R = self._generate_matrices()

        super().__init__()

    @property
    def name(self) -> str:
        """Name of the problem."""
        return "rand"

    def _construct_state_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """States are just indices from 0 to states-1."""
        return (
            jnp.zeros(1, dtype=jnp.int32),
            jnp.array([self.states - 1], dtype=jnp.int32),
        )

    def _construct_action_space(self) -> jnp.ndarray:
        """Actions are indices from 0 to actions-1."""
        return jnp.arange(self.actions)

    def action_components(self) -> list[str]:
        """Return list of action component names."""
        return ["action"]  # only one action component, the action

    def _construct_random_event_space(self) -> jnp.ndarray:
        """Random events are next states (0 to states-1)."""
        return jnp.arange(self.states).reshape(-1, 1)

    def random_event_probabilities(
        self, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Return transition probabilities for current state-action pair."""
        return self.P[action, state]

    def transition(
        self, state: jnp.ndarray, action: jnp.ndarray, random_event: jnp.ndarray
    ) -> tuple[jnp.ndarray, float]:
        """Compute next state and reward."""
        next_state = random_event[0]
        reward = self.R[action, state, next_state]
        return jnp.array([next_state]), reward

    def initial_values(self) -> jnp.ndarray:
        """Start with zeros."""
        return jnp.zeros(self.states)

    def _generate_matrices(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Generate random P and R matrices."""
        P = np.zeros((self.actions, self.states, self.states))
        R = np.zeros((self.actions, self.states, self.states))

        for action in range(self.actions):
            for state in range(self.states):
                # Create mask if not provided
                if self.mask is None:
                    m = np.random.random(self.states)
                    r = np.random.random()
                    m[m <= r] = 0
                    m[m > r] = 1
                elif self.mask.shape == (self.actions, self.states, self.states):
                    m = self.mask[action][state]
                else:
                    m = self.mask[state]

                # Ensure at least one transition
                if m.sum() == 0:
                    m[np.random.randint(0, self.states)] = 1

                # Generate probabilities and rewards
                P[action, state] = m * np.random.random(self.states)
                P[action, state] = P[action, state] / P[action, state].sum()
                R[action, state] = m * (2 * np.random.random(self.states) - 1)

        return jnp.array(P), jnp.array(R)
