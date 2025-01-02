"""Perishable inventory MDP problem from De Moor et al. (2022)."""

import jax
import jax.numpy as jnp
import numpyro.distributions
from hydra.conf import dataclass
from jaxtyping import Array, Float

from mdpax.core.problem import Problem, ProblemConfig
from mdpax.utils.spaces import (
    construct_space_from_bounds,
    space_dimensions_from_bounds,
    space_with_dimensions_to_index,
)
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
class DeMoorSingleProductPerishableConfig(ProblemConfig):
    """Configuration for the De Moor Perishable problem."""

    _target_: str = (
        "mdpax.problems.perishable_inventory.de_moor_single_product.DeMoorSingleProductPerishable"
    )
    max_demand: int = 100
    demand_gamma_mean: float = 4.0
    demand_gamma_cov: float = 0.5
    max_useful_life: int = 2
    lead_time: int = 1
    max_order_quantity: int = 10
    variable_order_cost: float = 3.0
    shortage_cost: float = 5.0
    wastage_cost: float = 7.0
    holding_cost: float = 1.0
    issue_policy: str = "lifo"


class DeMoorSingleProductPerishable(Problem):
    """Perishable inventory MDP problem from De Moor et al. (2022).

    Original paper: https://doi.org/10.1016/j.ejor.2021.10.045

    Models a single-product, single-echelon, periodic review perishable
    inventory replenishment problem where all stock has the same remaining
    useful life at arrival.

    State Space (state_dim = lead_time + max_useful_life - 1):
        Vector containing:
        - Orders in transit: [lead_time-1] elements in range [0, max_order_quantity]
        - Stock by age: [max_useful_life] elements in range [0, max_order_quantity],
          ordered with oldest units on the right

    Action Space (action_dim = 1):
        Vector containing:
        - Order quantity: 1 element in range [0, max_order_quantity]

    Random Events (event_dim = 1):
        Vector containing:
        - Demand: 1 element in range [0, max_demand]

    Dynamics:
        1. Place replenishment order
        2. Sample demand from truncated, discretized gamma distribution
        3. Issue stock using FIFO or LIFO policy
        4. Age remaining stock one period and discard expired units
        5. Reward is negative of total costs:
           - Variable ordering costs (per unit ordered)
           - Shortage costs (per unit of unmet demand)
           - Wastage costs (per unit that expires)
           - Holding costs (per unit in stock at end of period)
        6. Receive order placed lead_time - 1 periods ago immediately
            before the next period

    Args:
        max_demand: Maximum possible demand per period
        demand_gamma_mean: Mean of gamma distribution for demand
        demand_gamma_cov: Coefficient of variation of demand distribution
        max_useful_life: Number of periods before stock expires
        lead_time: Number of periods between order and delivery
        max_order_quantity: Maximum units that can be ordered
        variable_order_cost: Cost per unit ordered
        shortage_cost: Cost per unit of unmet demand
        wastage_cost: Cost per unit that expires
        holding_cost: Cost per unit held in stock at the end of each period
        issue_policy: Stock issuing policy ('fifo' or 'lifo')
    """

    def __init__(
        self,
        max_demand: int = 100,
        demand_gamma_mean: float = 4.0,
        demand_gamma_cov: float = 0.5,
        max_useful_life: int = 2,
        lead_time: int = 1,
        max_order_quantity: int = 10,
        variable_order_cost: float = 3.0,
        shortage_cost: float = 5.0,
        wastage_cost: float = 7.0,
        holding_cost: float = 1.0,
        issue_policy: str = "lifo",
    ):

        assert (
            max_useful_life >= 1
        ), "max_useful_life must be greater than or equal to 1"
        assert lead_time >= 1, "lead_time must be greater than or equal to 1"
        assert issue_policy in ["fifo", "lifo"], "Issue policy must be 'fifo' or 'lifo'"

        self.max_demand = max_demand
        self.demand_gamma_mean = demand_gamma_mean
        self.demand_gamma_cov = demand_gamma_cov
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time
        self.max_order_quantity = max_order_quantity
        self.cost_components = jnp.array(
            [
                variable_order_cost,
                shortage_cost,
                wastage_cost,
                holding_cost,
            ]
        )
        self.issue_policy = issue_policy
        if self.issue_policy == "fifo":
            self._issue_stock = self._issue_fifo
        elif self.issue_policy == "lifo":
            self._issue_stock = self._issue_lifo
        else:
            raise ValueError(f"Invalid issuing policy: {self.issue_policy}")

        super().__init__()

    @property
    def name(self) -> str:
        """Name of the problem."""
        return "de_moor_single_product"

    def _setup_before_space_construction(self):
        """Setup before space construction."""
        (
            self.demand_gamma_alpha,
            self.demand_gamma_beta,
        ) = self._convert_gamma_parameters(
            self.demand_gamma_mean, self.demand_gamma_cov
        )

        self.demand_probabilities = self._calculate_demand_probabilities(
            self.demand_gamma_alpha, self.demand_gamma_beta
        )

        self._state_dimensions = space_dimensions_from_bounds(self._state_bounds)
        self.state_component_lookup = self._construct_state_component_lookup()
        self.action_component_lookup = self._construct_action_component_lookup()
        self.random_event_component_lookup = (
            self._construct_random_event_component_lookup()
        )

    def _setup_after_space_construction(self):
        """Setup after space construction."""
        pass

    def _construct_state_space(self) -> StateSpace:
        """Construct state space.

        Returns:
            Array containing all possible states [n_states, state_dim]
        """
        return construct_space_from_bounds(self._state_bounds)

    def _construct_action_space(self) -> ActionSpace:
        """Construct action space.

        Returns:
            Array containing all possible actions [n_actions, action_dim]
        """
        return jnp.arange(0, self.max_order_quantity + 1).reshape(-1, 1)

    def _construct_random_event_space(self) -> RandomEventSpace:
        """Construct random event space.

        Returns:
            Array containing all possible random events [n_events, event_dim]
        """
        return jnp.arange(0, self.max_demand + 1).reshape(-1, 1)

    @property
    def _state_bounds(
        self,
    ) -> tuple[Float[Array, "state_dim"], Float[Array, "state_dim"]]:
        """Return min and max values for each state dimension.

        State dimensions are:
        - First (L-1) dimensions: in-transit orders [0, max_order_quantity]
        - Next M dimensions: stock at each age [0, max_order_quantity]

        Returns:
            Tuple of (mins, maxs) where each array has shape [state_dim]
            and state_dim = lead_time-1 + max_useful_life
        """
        state_dim = self.max_useful_life + self.lead_time - 1
        mins = jnp.zeros(state_dim, dtype=jnp.int32)
        maxs = jnp.full(state_dim, self.max_order_quantity, dtype=jnp.int32)
        return mins, maxs

    def state_to_index(self, state: StateVector) -> int:
        """Convert state vector to index.

        Args:
            state: State vector to convert [state_dim]

        Returns:
            Integer index of the state in state_space
        """
        return space_with_dimensions_to_index(state, self._state_dimensions)

    def random_event_probability(
        self,
        state: StateVector,
        action: ActionVector,
        random_event: RandomEventVector,
    ) -> float:
        """Compute probability of random event given state and action.

        Demand follows a discretized gamma distribution with mean demand_gamma_mean
        and coefficient of variation demand_gamma_cov. The demand distribution is
        independent of the current state and action.

        Args:
            state: Current state vector [state_dim]
            action: Action vector [action_dim]
            random_event: Random event vector [event_dim]

        Returns:
            Probability of this demand value occurring
        """
        return self.demand_probabilities[random_event]

    def transition(
        self, state: StateVector, action: ActionVector, random_event: RandomEventVector
    ) -> tuple[StateVector, Reward]:
        """Compute next state and reward for inventory transition.

        Processes one step of the perishable inventory system:
        1. Place replenishment order
        2. Sample demand from truncated, discretized gamma distribution
        3. Issue stock using FIFO or LIFO policy
        4. Age remaining stock one period and discard expired units
        5. Reward is negative of total costs:
           - Variable ordering costs (per unit ordered)
           - Shortage costs (per unit of unmet demand)
           - Wastage costs (per unit that expires)
           - Holding costs (per unit in stock at end of period)
        6. Receive order placed lead_time - 1 periods ago immediately
            before the next period

        Args:
            state: Current state vector [state_dim] containing:
                - Orders in transit [lead_time-1]
                - Current stock levels by age [max_useful_life]
            action: Action vector [action_dim] containing order quantity
            random_event: Random event vector [event_dim] containing demand

        Returns:
            Tuple of (next_state, reward) where:
                - next_state is the resulting state vector [state_dim]
                - reward is the negative of total costs for this step
        """
        demand = random_event[self.random_event_component_lookup["demand"]]
        opening_in_transit = state[self.state_component_lookup["in_transit"]]
        opening_stock = state[self.state_component_lookup["stock"]]

        in_transit = jnp.hstack([action, opening_in_transit])

        stock_after_issue = self._issue_stock(opening_stock, demand)

        # Compute variables required to calculate the cost
        variable_order = action[self.action_component_lookup["order_quantity"]]
        shortage = jnp.max(jnp.array([demand - jnp.sum(opening_stock), 0]))
        expiries = stock_after_issue[-1]
        holding = jnp.sum(stock_after_issue[0 : self.max_useful_life - 1])
        # These components must be in the same order as self.cost_components
        transition_function_reward_output = jnp.hstack(
            [variable_order, shortage, expiries, holding]
        )

        # Calculate single step reward
        reward = self._calculate_single_step_reward(
            state, action, transition_function_reward_output
        )

        # Age stock and Receive order placed at step t-(L-1)
        closing_stock = jnp.hstack(
            [in_transit[-1], stock_after_issue[0 : self.max_useful_life - 1]]
        )
        closing_in_transit = in_transit[0 : self.lead_time - 1]

        next_state = jnp.hstack([closing_in_transit, closing_stock]).astype(jnp.int32)

        return next_state, reward

    def get_problem_config(self) -> DeMoorSingleProductPerishableConfig:
        """Get problem configuration for reconstruction.

        Returns:
            Configuration containing all parameters needed to reconstruct
            this problem instance
        """
        return DeMoorSingleProductPerishableConfig(
            max_demand=int(self.max_demand),
            demand_gamma_mean=float(self.demand_gamma_mean),
            demand_gamma_cov=float(self.demand_gamma_cov),
            max_useful_life=int(self.max_useful_life),
            lead_time=int(self.lead_time),
            max_order_quantity=int(self.max_order_quantity),
            variable_order_cost=float(self.cost_components[0]),
            shortage_cost=float(self.cost_components[1]),
            wastage_cost=float(self.cost_components[2]),
            holding_cost=float(self.cost_components[3]),
            issue_policy=self.issue_policy,
        )

    # Transition function helper methods
    # ----------------------------------

    def _construct_state_component_lookup(self) -> dict[str, int | slice]:
        """Build mapping from state components to indices."""
        return {
            "in_transit": slice(0, self.lead_time - 1),
            "stock": slice(self.lead_time - 1, self.max_useful_life),
        }

    def _construct_action_component_lookup(self) -> dict[str, int | slice]:
        """Build mapping from action components to indices."""
        return {
            "order_quantity": 0,
        }

    def _construct_random_event_component_lookup(self) -> dict[str, int | slice]:
        """Build mapping from random event components to indices."""
        return {
            "demand": 0,
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
            Updated stock levels after issuing [max_useful_life]
        """
        _, remaining_stock = jax.lax.scan(
            self._issue_one_step, demand, opening_stock, reverse=True
        )
        return remaining_stock

    def _issue_lifo(
        self, opening_stock: Float[Array, "max_useful_life"], demand: int
    ) -> Float[Array, "max_useful_life"]:
        """Issue stock using LIFO (Last-In-First-Out) policy.

        Issues stock starting with newest items first (left side of vector).
        Uses scan to process each age category in sequence.

        Args:
            opening_stock: Current stock levels by age [max_useful_life]
            demand: Total customer demand to satisfy

        Returns:
            Updated stock levels after issuing [max_useful_life]
        """
        _, remaining_stock = jax.lax.scan(self._issue_one_step, demand, opening_stock)
        return remaining_stock

    def _issue_one_step(
        self, remaining_demand: int, stock_element: int
    ) -> tuple[int, int]:
        """Process one age category during stock issuing.

        Args:
            remaining_demand: Unfulfilled demand to satisfy
            stock_element: Available stock of current age

        Returns:
            Tuple of (remaining_demand, remaining_stock) where:
                - remaining_demand is unfulfilled demand after this age
                - remaining_stock is stock left in this age category
        """
        remaining_stock = (stock_element - remaining_demand).clip(0)
        remaining_demand = (remaining_demand - stock_element).clip(0)
        return remaining_demand, remaining_stock

    def _calculate_single_step_reward(
        self,
        state: StateVector,
        action: ActionVector,
        transition_function_reward_output: Float[Array, "4"],
    ) -> Reward:
        """Calculate reward (negative cost) for one transition step.

        Computes total cost by combining:
        - Variable ordering costs
        - Shortage costs
        - Wastage costs from expired items
        - Holding costs for inventory

        Args:
            state: Current state vector [state_dim]
            action: Action vector [action_dim]
            transition_function_reward_output: Output for each cost component [4]

        Returns:
            Negative of total cost for this step
        """
        cost = jnp.dot(transition_function_reward_output, self.cost_components)
        return -1 * cost

    # Random event probability helper methods
    # ---------------------------------------

    def _convert_gamma_parameters(self, mean: float, cov: float) -> tuple[float, float]:
        """Convert mean and CoV to gamma distribution parameters.

        Converts from mean and coefficient of variation (CoV) to the
        shape (alpha) and rate (beta) parameters used by numpyro.distributions.Gamma.

        Args:
            mean: Mean of the gamma distribution
            cov: Coefficient of variation

        Returns:
            Tuple of (alpha, beta) parameters for gamma distribution
        """
        alpha = 1 / (cov**2)
        beta = 1 / (mean * cov**2)
        return alpha, beta

    def _calculate_demand_probabilities(
        self, gamma_alpha: float, gamma_beta: float
    ) -> Float[Array, "max_demand_plus_one"]:
        """Calculate discretized demand probabilities from gamma distribution.

        Discretizes a gamma distribution by integrating between half-integers:
        P(demand=d) = P(d-0.5 < X ≤ d+0.5) for d>0
        P(demand=0) = P(X ≤ 0.5)

        Any probability mass beyond max_demand is added to P(demand=max_demand).

        Args:
            gamma_alpha: Shape parameter of gamma distribution
            gamma_beta: Rate parameter of gamma distribution

        Returns:
            Array of probabilities [max_demand + 1] where index i gives P(demand=i)
        """
        cdf = numpyro.distributions.Gamma(gamma_alpha, gamma_beta).cdf(
            jnp.hstack([0, jnp.arange(0.5, self.max_demand + 1.5)])
        )
        # Calculate P(d-0.5 < X ≤ d+0.5) using differences of CDF
        demand_probabilities = jnp.diff(cdf)

        # Add truncated probability mass to final demand level
        demand_probabilities = demand_probabilities.at[-1].add(
            1 - demand_probabilities.sum()
        )
        return demand_probabilities
