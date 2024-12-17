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
    
    # Define as class constants since they're the same for all instances
    ACTIONS = jnp.array([0, 1])
    OUTCOMES = jnp.array([0, 1])
    
    def __init__(self, S: int = 3, r1: float = 4.0, r2: float = 2.0, p: float = 0.1):
        self.S = S
        self.r1 = r1
        self.r2 = r2
        self.p = p
    
    def get_states(self) -> jnp.ndarray:
        """States are tree ages from 0 to S-1."""
        return jnp.array(jnp.arange(self.S), dtype=jnp.int32).reshape(-1,1)
    
    def get_state_names(self) -> dict[str, int]:
        """Only one state component: age."""
        return {"age": 0}
    
    def get_state_index(self, state: tuple) -> int:
        """Single state component, so just return the state."""
        return state[0].astype(jnp.int32)
    
    def get_actions(self) -> jnp.ndarray:
        """Return array of actions [wait=0, cut=1].
        
        Actions
        -------
        0 : wait
            Let the trees continue growing
        1 : cut
            Cut down the trees for immediate reward
        """
        return self.ACTIONS
    
    def get_action_labels(self) -> list[str]:
        """Return list of action component names."""
        return ["cut"] # only one action component, cut when 1
    
    def get_random_outcomes(self) -> jnp.ndarray:
        """Return array of outcomes [no_fire=0, fire=1].
        
        Outcomes
        --------
        0 : no_fire
            Forest continues normal growth
        1 : fire
            Forest burns down, resetting to age 0
        """
        return self.OUTCOMES
    
    def get_outcome_index(self, outcome: int) -> int:
        """Outcome is already an index."""
        return outcome
    
    def deterministic_transition(
        self,
        state: int,
        action: jnp.ndarray,
        outcome: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute next state and reward for forest transition."""
        is_cut = action == 1
        is_fire = outcome == 1
        
        # Compute reward - only get reward when cutting
        reward = jnp.where(
            is_cut,
            # Cut reward depends on tree age
            jnp.where(
                state == self.S - 1,
                self.r2,
                jnp.where(state == 0, 0.0, 1.0)
            ),
            # No reward for waiting
            jnp.where(
                state == self.S - 1,
                self.r1,
                0.0
            )
        )
        
        # Compute next state
        next_state = jnp.where(
            is_cut | is_fire,
            # Reset to age 0 if cut or fire
            0,
            # Otherwise increment age up to S-1
            jnp.minimum(state + 1, self.S - 1)
        ).astype(jnp.int32)
        
        return next_state, reward
    
    def get_outcome_probabilities(
        self,
        state: int,
        action: jnp.ndarray,
        possible_random_outcomes: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute probability of each outcome given state and action.
        
        When waiting:
            - No fire probability is 1 - p
            - Fire probability is p

        When cutting:
            - No fire probability is 1
            - Fire probability is 0
        """
        is_wait = action == 0
        
        # Compute probabilities for each outcome
        probs = jnp.where(
            is_wait,
            jnp.array([1-self.p, self.p]),
            jnp.array([1.0, 0.0]) # never a fire when cutting
        )

        return probs
    
    def initial_value(self) -> float:
        """Initial value estimate based on immediate cut reward."""
        return jnp.zeros(self.S)