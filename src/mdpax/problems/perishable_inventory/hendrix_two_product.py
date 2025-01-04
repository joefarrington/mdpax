"""Perishable inventory MDP problem from Hendrix et al. (2019)."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
from hydra.conf import dataclass
from jaxtyping import Array, Float

from mdpax.core.problem import Problem, ProblemConfig
from mdpax.utils.spaces import create_range_space
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
class HendrixTwoProductPerishableConfig(ProblemConfig):
    """Configuration for the HendrixTwoProductPerishable problem.

    Args:
        max_useful_life: Number of periods before stock expires. Must be >= 1.
        demand_poisson_mean_a: Mean of Poisson distribution for product A demand. Must be positive.
        demand_poisson_mean_b: Mean of Poisson distribution for product B demand. Must be positive.
        substitution_probability: Probability of substituting A for B when B is out. Must be in [0,1].
        variable_order_cost_a: Cost per unit of product A ordered
        variable_order_cost_b: Cost per unit of product B ordered
        sales_price_a: Revenue per unit of product A sold
        sales_price_b: Revenue per unit of product B sold
        max_order_quantity_a: Maximum units of product A that can be ordered. Must be positive.
        max_order_quantity_b: Maximum units of product B that can be ordered. Must be positive.

    Example:
        >>> config = HendrixTwoProductPerishableConfig(
        ...     max_useful_life=3,
        ...     demand_poisson_mean_a=4.0,
        ...     max_order_quantity_a=15,
        ... )
        >>> problem = HendrixTwoProductPerishable(config=config)

        # Or using kwargs:
        >>> problem = HendrixTwoProductPerishable(max_useful_life=3)
    """

    _target_: str = (
        "mdpax.problems.perishable_inventory.hendrix_two_product.HendrixTwoProductPerishable"
    )
    max_useful_life: int = 2
    demand_poisson_mean_a: float = 5.0
    demand_poisson_mean_b: float = 5.0
    substitution_probability: float = 0.5
    variable_order_cost_a: float = 0.5
    variable_order_cost_b: float = 0.5
    sales_price_a: float = 1.0
    sales_price_b: float = 1.0
    max_order_quantity_a: int = 10
    max_order_quantity_b: int = 10

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_useful_life < 1:
            raise ValueError("max_useful_life must be greater than or equal to 1")
        if self.demand_poisson_mean_a <= 0:
            raise ValueError("demand_poisson_mean_a must be positive")
        if self.demand_poisson_mean_b <= 0:
            raise ValueError("demand_poisson_mean_b must be positive")
        if not 0 <= self.substitution_probability <= 1:
            raise ValueError("substitution_probability must be between 0 and 1")
        if self.max_order_quantity_a <= 0:
            raise ValueError("max_order_quantity_a must be positive")
        if self.max_order_quantity_b <= 0:
            raise ValueError("max_order_quantity_b must be positive")


class HendrixTwoProductPerishable(Problem):
    """Two-product perishable inventory MDP problem from Hendrix et al. (2019).

    Models a two-product, single-echelon, periodic review perishable
    inventory replenishment problem where all stock has the same remaining
    useful life at arrival and there is the possibility for substution between
    products.

    State Space (state_dim = 2 * max_useful_life):
        Vector containing:
            - Product A stock by age: [max_useful_life] elements in range [0, max_order_quantity_a], ordered with oldest units on the right
            - Product B stock by age: [max_useful_life] elements in range [0, max_order_quantity_b], ordered with oldest units on the right

    Action Space (action_dim = 2):
        Vector containing:
            - Product A order quantity: 1 element in range [0, max_order_quantity_a]
            - Product B order quantity: 1 element in range [0, max_order_quantity_b]

    Random Events (event_dim = 2):
        Vector containing:
            - Product A units issued: 1 element in range [0, max_stock_a]
            - Product B units issued: 1 element in range [0, max_stock_b]

    Dynamics:
        1. Place replenishment order
        2. Random event determines units issued of each product, incorporating both:
            - Poisson-distributed demand for each product
            - Possible substitution from A to B when B's demand exceeds stock
        3. Issue stock using FIFO policy for each product
        4. Age remaining stock one period and discard expired units
        5. Reward is revenue from units issued less variable ordering costs
        6. Receive order placed at the start of the period immediately before the next period

    Args:
        config: Configuration object. If provided, keyword arguments are ignored.
        **kwargs: Parameters matching :class:`HendrixTwoProductPerishableConfig`.
            See Config class for detailed parameter descriptions.


    References:
        - Hendrix et al. (2019): https://doi.org/10.1002/cmm4.1027

    Note:
        The three random elements in the transition are the demand for each product and
        the number of units of demand for product B willing to accept substitution with
        product A. For consistency with the original implementation, the random events
        are taken the be the number of units of each product issued. The transition is
        deterministic given the number of products of each type issued.
    """

    Config = HendrixTwoProductPerishableConfig

    def __init__(
        self, config: HendrixTwoProductPerishableConfig | None = None, **kwargs
    ):
        if config is not None:
            self.config = config
        else:
            self.config = self.Config(**kwargs)

        self.max_useful_life = self.config.max_useful_life
        self.demand_poisson_mean_a = self.config.demand_poisson_mean_a
        self.demand_poisson_mean_b = self.config.demand_poisson_mean_b
        self.substitution_probability = self.config.substitution_probability
        self.variable_order_cost_a = self.config.variable_order_cost_a
        self.variable_order_cost_b = self.config.variable_order_cost_b
        self.sales_price_a = self.config.sales_price_a
        self.sales_price_b = self.config.sales_price_b
        self.variable_order_costs = jnp.array(
            [self.variable_order_cost_a, self.variable_order_cost_b]
        )
        self.sales_prices = jnp.array([self.sales_price_a, self.sales_price_b])
        self.max_order_quantity_a = self.config.max_order_quantity_a
        self.max_order_quantity_b = self.config.max_order_quantity_b

        super().__init__()

    @property
    def name(self) -> str:
        """A unique identifier for this problem type"""
        return "hendrix_two_product"

    def _setup_before_space_construction(self) -> None:
        """Setup before space construction."""
        # Compute dynamic limits on stock and demand
        self.max_stock_a = self.max_order_quantity_a * self.max_useful_life
        self.max_stock_b = self.max_order_quantity_b * self.max_useful_life
        self.max_demand = self.max_useful_life * (
            max(self.max_order_quantity_a, self.max_order_quantity_b) + 2
        )

        # Build lookup tables for state, action, and random event components
        # so they can be used to index into state, action, and random event vectors
        # by name in transition function
        self.state_component_lookup = self._construct_state_component_lookup()
        self.action_component_lookup = self._construct_action_component_lookup()
        self.random_event_component_lookup = (
            self._construct_random_event_component_lookup()
        )

    def _setup_after_space_construction(self) -> None:
        """Setup after space construction."""
        # Precompute conditional probabilities
        self.pu = self._calculate_pu()
        self.pz = self._calculate_pz()

    def _construct_state_space(self) -> StateSpace:
        """Build array of all possible states.

        Returns:
            Array of shape [n_states, state_dim] containing all possible states
        """

        mins = np.zeros(2 * self.max_useful_life, dtype=np.int32)
        maxs = np.hstack(
            [
                np.full(
                    self.max_useful_life,
                    self.max_order_quantity_a,
                ),
                np.full(
                    self.max_useful_life,
                    self.max_order_quantity_b,
                ),
            ]
        )
        state_space, self._state_to_index_fn = create_range_space(mins, maxs)
        return state_space

    def _construct_action_space(self) -> ActionSpace:
        """Build array of all possible actions.

        Returns:
            Array of shape [n_actions, action_dim] containing all possible actions
        """
        mins = np.array([0, 0])
        maxs = np.array([self.max_order_quantity_a, self.max_order_quantity_b])
        action_space, _ = create_range_space(mins, maxs)
        return action_space

    def _construct_random_event_space(self) -> RandomEventSpace:
        """Build array of all possible random events.

        Returns:
            Array of shape [n_events, event_dim] containing all possible random events
        """
        mins = np.array([0, 0])
        maxs = np.array([self.max_stock_a, self.max_stock_b])
        random_event_space, self._random_event_to_index = create_range_space(mins, maxs)
        return random_event_space

    def state_to_index(self, state: StateVector) -> int:
        """Convert state vector to index.

        Args:
            state: State vector to convert [state_dim]

        Returns:
            Integer index of the state in state_space
        """
        return self._state_to_index_fn(state)

    def random_event_probability(
        self,
        state: StateVector,
        action: ActionVector,
        random_event: RandomEventVector,
    ) -> float:
        """Compute probability of random event given state and action.

        The number of units issued of each product follows a compound distribution:
            - Demand for each product is Poisson distributed
            - For product B, if demand exceeds stock, excess demand can be satisfied by product A with binomial probability

        Args:
            state: Current state vector [state_dim]
            action: Action vector [action_dim]
            random_event: Random event vector [event_dim]

        Returns:
            Probability of this combination of issued units for both products
        """
        stock_a = jnp.sum(state[self.state_component_lookup["stock_a"]])
        stock_b = jnp.sum(state[self.state_component_lookup["stock_b"]])

        # Issued a less than stock of a, issued b less than stock of b
        probs_1 = self._get_probs_ia_lt_stock_a_ib_lt_stock_b(stock_a, stock_b)
        # Issued a equal to stock of a, issued b less than stock of b
        probs_2 = self._get_probs_ia_eq_stock_a_ib_lt_stock_b(stock_a, stock_b)
        # Issued a less than stock of a, issued b equal to stock of b
        probs_3 = self._get_probs_ia_lt_stock_a_ib_eq_stock_b(stock_a, stock_b)
        # Issued a equal to stock of a, issued b equal to stock of b
        probs_4 = self._get_probs_ia_eq_stock_a_ib_eq_stock_b(stock_a, stock_b)
        all_probs = (probs_1 + probs_2 + probs_3 + probs_4).reshape(-1)
        return all_probs[self._random_event_to_index(random_event)]

    def transition(
        self,
        state: StateVector,
        action: ActionVector,
        random_event: RandomEventVector,
    ) -> tuple[StateVector, Reward]:
        """Compute next state and reward for a transition.

        Processes one step of the two-product perishable inventory system:
            1. Place replenishment order
            2. Random event determines units issued of each product, incorporating both:
                - Poisson-distributed demand for each product
                - Possible substitution from A to B when B's demand exceeds stock
            3. Issue stock using FIFO policy for each product
            4. Age remaining stock one period and discard expired units
            5. Reward is revenue from units issued less variable ordering costs
            6. Receive order placed at the start of the period immediately before the next period

        Args:
            state: Current state vector [state_dim]
            action: Action vector [action_dim]
            random_event: Random event vector [event_dim]

        Returns:
            Tuple containing the next state vector [state_dim] and
            the immediate reward
        """
        issued_a = random_event[self.random_event_component_lookup["issued_a"]]
        issued_b = random_event[self.random_event_component_lookup["issued_b"]]

        opening_stock_a = state[self.state_component_lookup["stock_a"]]
        opening_stock_b = state[self.state_component_lookup["stock_b"]]

        stock_after_issue_a = self._issue_fifo(opening_stock_a, issued_a)
        stock_after_issue_b = self._issue_fifo(opening_stock_b, issued_b)

        # Pass through the random outcome (units issued)
        single_step_reward = self._calculate_single_step_reward(
            state, action, random_event
        )

        # Age stock one day and receive the order from the morning
        closing_stock_a = jnp.hstack(
            [
                action[self.action_component_lookup["order_quantity_a"]],
                stock_after_issue_a[0 : self.max_useful_life - 1],
            ]
        )
        closing_stock_b = jnp.hstack(
            [
                action[self.action_component_lookup["order_quantity_b"]],
                stock_after_issue_b[0 : self.max_useful_life - 1],
            ]
        )

        next_state = jnp.concatenate([closing_stock_a, closing_stock_b], axis=-1)

        return (
            next_state,
            single_step_reward,
        )

    def initial_value(self, state: StateVector) -> float:
        """Return initial value estimate for a given state.

        Initial value estimate based on one-step ahead expected sales revenue.

        Args:
            state: State vector [state_dim]

        Returns:
            Initial value estimate for the given state
        """
        return self._calculate_expected_sales_revenue(state)

    # Transition function helper methods
    # ----------------------------------

    def _construct_state_component_lookup(self) -> dict[str, int | slice]:
        """Build mapping from state components to indices."""
        m = self.max_useful_life
        return {
            "stock_a": slice(0, m),
            "stock_b": slice(m, 2 * m),
        }

    def _construct_action_component_lookup(self) -> dict[str, int]:
        """Build mapping from action components to indices."""
        return {
            "order_quantity_a": 0,
            "order_quantity_b": 1,
        }

    def _construct_random_event_component_lookup(self) -> dict[str, int]:
        """Build mapping from random event components to indices."""
        return {
            "issued_a": 0,
            "issued_b": 1,
        }

    def _issue_fifo(
        self, opening_stock: Float[Array, "max_useful_life"], demand: int
    ) -> Float[Array, "max_useful_life"]:
        """Issue stock using FIFO (First-In-First-Out) policy.

        Issues stock starting with oldest items first (right side of vector).
        Uses scan to process each age category in sequence.

        Args:
            opening_stock: Current stock levels by age [max_useful_life]
            demand: Total customer demand to satisfy

        Returns:
            Array of ppdated stock levels after issuing [max_useful_life]
        """
        _, remaining_stock = jax.lax.scan(
            self._issue_one_step, demand, opening_stock, reverse=True
        )
        return remaining_stock

    def _issue_one_step(
        self, remaining_demand: int, stock_element: int
    ) -> tuple[int, int]:
        """Process one age category during stock issuing.

        Args:
            remaining_demand: Unfulfilled demand to satisfy
            stock_element: Available stock of current age

        Returns:
            Tuple containing the remaining demand and remaining stock
            after processing this age category
        """
        remaining_stock = (stock_element - remaining_demand).clip(0)
        remaining_demand = (remaining_demand - stock_element).clip(0)
        return remaining_demand, remaining_stock

    def _calculate_single_step_reward(
        self,
        state: StateVector,
        action: ActionVector,
        random_event: RandomEventVector,
    ) -> Reward:
        """Calculate reward (revenue minus costs) for one transition step.

        Computes total reward by combining:
        - Variable ordering costs (negative)
        - Sales revenue from issued stock (positive)

        Args:
            state: Current state vector [state_dim]
            action: Action vector [action_dim]
            random_event: Random event vector [event_dim] containing units issued

        Returns:
            Revenue minus costs for this step
        """
        cost = jnp.dot(action, self.variable_order_costs)
        revenue = jnp.dot(random_event, self.sales_prices)
        return revenue - cost

    # Random event probability helper methods
    # ---------------------------------------

    def _calculate_pu(self) -> Float[Array, "max_demand_plus_one max_stock_b_plus_one"]:
        """Calculate conditional probabilities for substitution demand.

        Returns:
            Array of probabilities where pu[u,y] is Prob(u|y),
            the conditional probability of u substitution demand given y units
            of product B in stock. Shape is [max_demand + 1, max_stock_b + 1].
        """
        pu = np.zeros((self.max_demand + 1, self.max_stock_b + 1))
        for y in range(0, self.max_stock_b + 1):
            x = np.arange(0, self.max_demand - y)
            pu[0, y] = scipy.stats.poisson.pmf(x + y, self.demand_poisson_mean_b).dot(
                scipy.stats.binom.pmf(0, x, self.substitution_probability)
            )

            for u in range(1, self.max_demand - y):
                x = np.arange(u, self.max_demand - y)
                pu[u, y] = scipy.stats.poisson.pmf(
                    x + y, self.demand_poisson_mean_b
                ).dot(scipy.stats.binom.pmf(u, x, self.substitution_probability))

        return jnp.array(pu)

    def _calculate_pz(self) -> Float[Array, "max_demand_plus_one max_stock_b_plus_one"]:
        """Calculate conditional probabilities for total demand for product A.

        Returns:
            Array of probabilities where pz[z,y] is Prob(z|y),
            the conditional probability of z total demand for product A given
            demand for product B is at least equal to y units in stock.
            Shape is [max_demand + 1, max_stock_b + 1].
        """
        pz = np.zeros((self.max_demand + 1, self.max_stock_b + 1))
        pa = scipy.stats.poisson.pmf(
            np.arange(self.max_demand + 1), self.demand_poisson_mean_a
        )
        # No demand for a itself, and no subst demand
        pz[0, :] = pa[0] * self.pu[0, :]
        for y in range(0, self.max_stock_b + 1):
            for z in range(1, self.max_demand + 1):
                pz[z, y] = pa[np.arange(0, z + 1)].dot(
                    self.pu[z - np.arange(0, z + 1), y]
                )
        return jnp.array(pz)

    def _get_probs_ia_lt_stock_a_ib_lt_stock_b(
        self, stock_a: chex.Array, stock_b: chex.Array
    ) -> chex.Array:
        """Calculate probabilities for case where issued quantities are below stock levels."""
        # P(i_a, i_b) = P(d_a=ia) * P(d_b=ib)
        # Easy cases, all demand met and no substitution
        prob_da = jax.scipy.stats.poisson.pmf(
            jnp.arange(self.max_stock_a + 1), self.demand_poisson_mean_a
        )
        prob_da_masked = prob_da * (jnp.arange(self.max_stock_a + 1) < stock_a)
        prob_db = jax.scipy.stats.poisson.pmf(
            jnp.arange(self.max_stock_b + 1), self.demand_poisson_mean_b
        )
        prob_db_masked = prob_db * (jnp.arange(self.max_stock_b + 1) < stock_b)
        issued_probs = jnp.outer(prob_da_masked, prob_db_masked)

        return issued_probs

    def _get_probs_ia_eq_stock_a_ib_lt_stock_b(
        self, stock_a: chex.Array, stock_b: chex.Array
    ) -> chex.Array:
        """Calculate probabilities for case where product A issued equals stock level."""
        # Therefore P(i_a, i_b) = P(d_a>=ia) * P(d_b=ib)
        # No substitution
        issued_probs = jnp.zeros((self.max_stock_a + 1, self.max_stock_b + 1))

        # Demand for a higher than stock_a, but demand for b less than than stock_b
        prob_da_gteq_stock_a = 1 - jax.scipy.stats.poisson.cdf(
            stock_a - 1, self.demand_poisson_mean_a
        )
        prob_db = jax.scipy.stats.poisson.pmf(
            jnp.arange(self.max_stock_b + 1), self.demand_poisson_mean_b
        )
        prob_db_masked = prob_db * (jnp.arange(self.max_stock_b + 1) < stock_b)
        probs = prob_da_gteq_stock_a * prob_db_masked
        issued_probs = issued_probs.at[stock_a, :].add(probs)

        return issued_probs

    def _get_probs_ia_lt_stock_a_ib_eq_stock_b(
        self, stock_a: chex.Array, stock_b: chex.Array
    ) -> chex.Array:
        """Calculate probabilities for case where product B issued equals stock level."""
        # Therefore total demand for a is < stock_a, demand for b >= stock_b
        issued_probs = jnp.zeros((self.max_stock_a + 1, self.max_stock_b + 1))

        # Demand for b higher than stock_b, so substitution possible

        probs_issued_a = jax.lax.dynamic_slice(
            self.pz, (0, stock_b), (self.max_demand + 1, 1)
        ).reshape(-1)

        probs_issued_a_masked = probs_issued_a * (
            jnp.arange(len(probs_issued_a)) < stock_a
        )

        # Trim array to max_stock_a
        probs_issued_a_masked = jax.lax.dynamic_slice(
            probs_issued_a_masked, (0,), (self.max_stock_a + 1,)
        )

        issued_probs = issued_probs.at[:, stock_b].add(probs_issued_a_masked)

        return issued_probs

    def _get_probs_ia_eq_stock_a_ib_eq_stock_b(
        self, stock_a: chex.Array, stock_b: chex.Array
    ) -> chex.Array:
        """Calculate probabilities for case where both products issued equal stock levels."""
        # Therefore total demand for a is >= stock_a, demand for b >= stock_b
        issued_probs = jnp.zeros((self.max_stock_a + 1, self.max_stock_b + 1))

        # Demand for b higher than stock_b, so subsitution possible
        probs_issued_a = jax.lax.dynamic_slice(
            self.pz, (0, stock_b), (self.max_demand + 1, 1)
        ).reshape(-1)
        prob_combined_demand_gteq_stock_a = probs_issued_a.dot(
            jnp.arange(len(probs_issued_a)) >= stock_a
        )

        issued_probs = issued_probs.at[stock_a, stock_b].add(
            prob_combined_demand_gteq_stock_a
        )

        return issued_probs

    # Initial value helper methods
    # ---------------------------------------

    def _calculate_sales_revenue_for_possible_random_events(
        self,
    ) -> Float[Array, "n_events"]:
        """Calculate the sales revenue for each possible random event.

        Returns:
            Array of sales revenue for each possible random event [n_events]
        """
        return (self.random_event_space.dot(self.sales_prices)).reshape(-1)

    def _calculate_expected_sales_revenue(self, state: StateVector) -> float:
        """Calculate the expected sales revenue for a given state.

        Args:
            state: State vector to calculate expected revenue for [state_dim]

        Returns:
            Expected sales revenue for one step from this state
        """
        issued_probabilities = jax.vmap(
            self.random_event_probability, in_axes=(None, None, 0)
        )(state, 0, self.random_event_space)
        expected_sales_revenue = issued_probabilities.dot(
            self._calculate_sales_revenue_for_possible_random_events()
        )
        return expected_sales_revenue
