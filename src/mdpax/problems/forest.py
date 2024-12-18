"""Forest management MDP problem."""

import jax.numpy as jnp

from mdpax.core.problem import Problem


class Forest(Problem):
    """Forest management MDP.

    The forest management problem involves deciding whether to cut down trees
    for immediate reward or wait for them to grow larger. There is a risk of
    fire destroying the forest, which increases with tree size.

    Parameters
    ----------
    S : int
        Number of states (tree ages from 0 to S-1)
    r1 : float
        Reward for cutting young trees
    r2 : float
        Additional reward for cutting mature trees
    p : float
        Base probability of fire
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

    def _construct_state_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return min and max values for each state dimension.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            (mins, maxs) where each is shape [n_dims] and n_dims = 1
        """
        n_dims = 1

        # Single dimension by [0, S-1]
        mins = jnp.zeros(n_dims, dtype=jnp.int32)
        maxs = jnp.full(n_dims, self.S - 1, dtype=jnp.int32)

        return mins, maxs

    def _construct_action_space(self) -> jnp.ndarray:
        """Return array of actions [wait=0, cut=1].

        Actions
        -------
        0 : wait
            Let the trees continue growing
        1 : cut
            Cut down the trees for immediate reward
        """
        return jnp.array([0, 1])

    def action_components(self) -> list[str]:
        """Return list of action component names."""
        return ["cut"]  # only one action component, cut when 1

    def _construct_random_event_space(self) -> jnp.ndarray:
        """Return array of outcomes [no_fire=0, fire=1].

        Outcomes
        --------
        0 : no_fire
            Forest continues normal growth
        1 : fire
            Forest burns down, resetting to age 0
        """
        return jnp.array([0, 1])

    def random_event_probability(
        self, state: int, action: jnp.ndarray, random_event: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute probability of each outcome given state and action.

        When waiting:
            - No fire probability is 1 - p
            - Fire probability is p

        When cutting:
            - No fire probability is 1
            - Fire probability is 0
        """
        return self.probability_matrix[action, random_event]

    def transition(
        self, state: int, action: jnp.ndarray, random_event: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute next state and reward for forest transition."""
        is_cut = action == 1
        is_fire = random_event == 1

        # Compute reward - only get reward when cutting
        reward = jnp.where(
            is_cut,
            # Cut reward depends on tree age
            jnp.where(state == self.S - 1, self.r2, jnp.where(state == 0, 0.0, 1.0)),
            # No reward for waiting
            jnp.where(state == self.S - 1, self.r1, 0.0),
        )

        # Compute next state
        next_state = jnp.where(
            is_cut | is_fire,
            # Reset to age 0 if cut or fire
            0,
            # Otherwise increment age up to S-1
            jnp.minimum(state + 1, self.S - 1),
        ).astype(jnp.int32)

        return next_state, reward

    def initial_values(self) -> float:
        """Initial value estimate based on immediate cut reward."""
        return jnp.zeros(self.n_states)
