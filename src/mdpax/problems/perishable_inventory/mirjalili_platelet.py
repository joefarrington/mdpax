"""Perishable inventory MDP problem from Mirjalili (2022)."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
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
class MirjaliliPlateletPerishableConfig(ProblemConfig):
    """Configuration for the MirjaliliPlateletPerishable problem.

    Args:
        max_demand: Maximum possible demand per period. Must be positive.
        weekday_demand_negbin_n: Parameter n of negative binomial distribution for each weekday,
            [M, T, W, T, F, S, S]. All values must be positive.
        weekday_demand_negbin_delta: Parameter delta of negative binomial distribution for each weekday,
            [M, T, W, T, F, S, S]. All values must be positive.
        max_useful_life: Number of periods before stock expires. Must be >= 1.
        useful_life_at_arrival_distribution_c_0: Base logit parameters for useful life at arrival,
            [2, ..., max_useful_life]. Length must be max_useful_life - 1.
        useful_life_at_arrival_distribution_c_1: Order quantity multiplier for useful life logits,
            [2, ..., max_useful_life]. Length must be max_useful_life - 1.
        max_order_quantity: Maximum units that can be ordered. Must be positive.
        variable_order_cost: Cost per unit ordered
        fixed_order_cost: Cost incurred when order > 0
        shortage_cost: Cost per unit of unmet demand
        wastage_cost: Cost per unit that expires
        holding_cost: Cost per unit held in stock at end of period

    Example:
        >>> config = MirjaliliPlateletPerishableConfig(
        ...     max_demand=30,
        ...     max_useful_life=4,
        ...     max_order_quantity=25,
        ... )
        >>> problem = MirjaliliPlateletPerishable(config=config)

        # Or using kwargs:
        >>> problem = MirjaliliPlateletPerishable(max_demand=30)
    """

    _target_: str = (
        "mdpax.problems.perishable_inventory.mirjalili_platelet.MirjaliliPlateletPerishable"
    )
    max_demand: int = 20
    weekday_demand_negbin_n: tuple[float, ...] = (3.5, 11.0, 7.2, 11.1, 5.9, 5.5, 2.2)
    weekday_demand_negbin_delta: tuple[float, ...] = (5.7, 6.9, 6.5, 6.2, 5.8, 3.3, 3.4)
    max_useful_life: int = 3
    useful_life_at_arrival_distribution_c_0: tuple[float, ...] = (1.0, 0.5)
    useful_life_at_arrival_distribution_c_1: tuple[float, ...] = (0.0, 0.0)
    max_order_quantity: int = 20
    variable_order_cost: float = 0.0
    fixed_order_cost: float = 10.0
    shortage_cost: float = 20.0
    wastage_cost: float = 5.0
    holding_cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_demand <= 0:
            raise ValueError("max_demand must be positive")
        if len(self.weekday_demand_negbin_n) != 7:
            raise ValueError("weekday_demand_negbin_n must have length 7")
        if any(n <= 0 for n in self.weekday_demand_negbin_n):
            raise ValueError("all weekday_demand_negbin_n values must be positive")
        if len(self.weekday_demand_negbin_delta) != 7:
            raise ValueError("weekday_demand_negbin_delta must have length 7")
        if any(d <= 0 for d in self.weekday_demand_negbin_delta):
            raise ValueError("all weekday_demand_negbin_delta values must be positive")
        if self.max_useful_life < 1:
            raise ValueError("max_useful_life must be greater than or equal to 1")
        if (
            len(self.useful_life_at_arrival_distribution_c_0)
            != self.max_useful_life - 1
        ):
            raise ValueError(
                "useful_life_at_arrival_distribution_c_0 must have length max_useful_life - 1"
            )
        if (
            len(self.useful_life_at_arrival_distribution_c_1)
            != self.max_useful_life - 1
        ):
            raise ValueError(
                "useful_life_at_arrival_distribution_c_1 must have length max_useful_life - 1"
            )
        if self.max_order_quantity <= 0:
            raise ValueError("max_order_quantity must be positive")


WEEKDAYS = [
    "Monday",  # idx 0
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",  # idx 6
]


class MirjaliliPlateletPerishable(Problem):
    """Platelet inventory MDP problem from Mirjalili (2022).

    Models a single-product, single-echelon, periodic review perishable
    inventory replenishment problem for platelets in a hospital blood bank
    where the products have a fixed maximum useful life but uncertain remaining
    useful life at arrival. The distribution of remaining useful life at arrival
    may depend on the order quantity.

    State Space (state_dim = max_useful_life):
        Vector containing:
            - Weekday: 1 element in range [0, 6] (Monday to Sunday)
            - Stock by age: [max_useful_life-1] elements in range [0, max_order_quantity], ordered with oldest units on the right

    Action Space (action_dim = 1):
        Vector containing:
            - Order quantity: 1 element in range [0, max_order_quantity]

    Random Events (event_dim = max_useful_life + 1):
        Vector containing:
            - Demand: 1 element in range [0, max_demand]
            - Stock received by age: [max_useful_life] elements in range [0, max_order_quantity] summing to at most max_order_quantity

    Dynamics:
        1. Place replenishment order
        2. Immediately receive the order, where the remaining useful life of the units at arrival is sampled from a multinomial distribution with parameters that may depend on the order quantity
        3. Sample demand from weekday-specific truncated negative binomial distribution
        4. Issue stock using OUFO (Oldest Units First Out) policy
        5. Age remaining stock one period and discard expired units
        6. Reward is negative of total costs:
            - Variable ordering costs (per unit ordered)
            - Fixed ordering costs (when order > 0)
            - Shortage costs (per unit of unmet demand)
            - Wastage costs (per unit that expires)
            - Holding costs (per unit in stock at end of period, including expiring units)
        7. Update weekday to next day of week

    Args:
        config: Configuration object. If provided, keyword arguments are ignored.
        **kwargs: Parameters matching :class:`MirjaliliPlateletPerishableConfig`.
            See Config class for detailed parameter descriptions.

    References:
        - Mirjalili (2022): https://tspace.library.utoronto.ca/bitstream/1807/124976/1/Mirjalili_Mahdi_202211_PhD_thesis.pdf
        - Abouee-Mehrizi et al. (2023): https://doi.org/10.48550/arXiv.2307.09395

    Note:
        - In the original source, the demand distribution is a truncated negative
          binomial distribution over the number of failured before reaching a specified
          number of successed parameterized by n (target number of successes)
          and delta (expected value).
        - The probability of success of a trial is n/(n + delta).
    """

    Config = MirjaliliPlateletPerishableConfig

    def __init__(
        self, config: MirjaliliPlateletPerishableConfig | None = None, **kwargs
    ):
        if config is not None:
            self.config = config
        else:
            self.config = self.Config(**kwargs)

        self.max_demand = self.config.max_demand
        self.useful_life_at_arrival_distribution_c_0 = jnp.array(
            self.config.useful_life_at_arrival_distribution_c_0
        )
        self.useful_life_at_arrival_distribution_c_1 = jnp.array(
            self.config.useful_life_at_arrival_distribution_c_1
        )
        self.weekday_demand_negbin_n = jnp.array(self.config.weekday_demand_negbin_n)
        self.weekday_demand_negbin_delta = jnp.array(
            self.config.weekday_demand_negbin_delta
        )
        self.max_useful_life = self.config.max_useful_life
        self.max_order_quantity = self.config.max_order_quantity
        self.cost_components = jnp.array(
            [
                self.config.variable_order_cost,
                self.config.fixed_order_cost,
                self.config.shortage_cost,
                self.config.wastage_cost,
                self.config.holding_cost,
            ]
        )

        super().__init__()

    @property
    def name(self) -> str:
        """A unique identifier for this problem type"""
        return "mirjalili_platelet"

    def _setup_before_space_construction(self) -> None:
        """Setup before space construction."""
        # Calculate probability of success, from parameterisation provided in MM thesis
        # to parameters for numpyro.distributions.NegativeBinomial
        self.weekday_demand_negbin_p = self.weekday_demand_negbin_n / (
            self.weekday_demand_negbin_delta + self.weekday_demand_negbin_n
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
        pass

    def _construct_state_space(self) -> StateSpace:
        """Build array of all possible states.

        Returns:
            Array of shape [n_states, state_dim] containing all possible states
        """

        mins = np.zeros(self.max_useful_life, dtype=np.int32)
        maxs = np.hstack(
            [
                np.array([6]),  # weekday
                np.full(
                    self.max_useful_life - 1,
                    self.max_order_quantity,
                ),  # stock
            ]
        )
        state_space, self._state_to_index_fn = create_range_space(mins, maxs)
        return state_space

    def _construct_action_space(self) -> ActionSpace:
        """Build array of all possible actions.

        Returns:
            Array of shape [n_actions, action_dim] containing all possible actions
        """
        return jnp.arange(0, self.max_order_quantity + 1).reshape(-1, 1)

    def _construct_random_event_space(self) -> RandomEventSpace:
        """Build array of all possible random events.

        Returns:
            Array of shape [n_events, event_dim] containing all possible random events
        """
        demands = np.arange(self.max_demand + 1).reshape(1, -1)

        # Generate all possible combinations of received order quantities split by age
        rec_combinations = np.array(
            list(
                itertools.product(
                    *[
                        range(self.max_order_quantity + 1)
                        for _ in range(self.max_useful_life)
                    ]
                )
            )
        )

        # Filter out combinations where the total received exceeds max_order_quantity
        valid_rec_combinations = rec_combinations[
            rec_combinations.sum(axis=1) <= self.max_order_quantity
        ]

        # Repeat demands for each valid combination and stack them
        repeated_demands = np.repeat(
            demands, len(valid_rec_combinations), axis=0
        ).reshape(-1, 1)
        repeated_valid_rec_combinations = np.repeat(
            valid_rec_combinations, self.max_demand + 1, axis=0
        )

        # Combine the two random elements - demand and remaining useful life on arrival
        return jnp.array(np.hstack([repeated_demands, repeated_valid_rec_combinations]))

    def state_to_index(self, state: StateVector) -> int:
        """Convert state vector to index.

        Args:
            state: State vector to convert [state_dim]

        Returns:
            Index of the state in state_space
        """
        return self._state_to_index_fn(state)

    def random_event_probability(
        self,
        state: StateVector,
        action: ActionVector,
        random_event: RandomEventVector,
    ) -> float:
        """Compute probability of random event given state and action.

        Combines demand probabilities (based on weekday) with order receipt
        probabilities (based on action).

        Args:
            state: Current state vector [state_dim]
            action: Action vector [action_dim]
            random_event: Random event vector [event_dim]

        Returns:
            Probability of this combination of demand and received stock
        """
        weekday = state[self.state_component_lookup["weekday"]]

        # Get probabilities for demand component
        demand_probs = self._calculate_demand_probabilities(weekday)
        demand_prob = demand_probs[
            random_event[self.random_event_component_lookup["demand"]]
        ]

        # Get probabilities for received order component
        received_prob = self._calculate_received_order_probabilities(
            action, random_event[self.random_event_component_lookup["stock_received"]]
        )

        return demand_prob * received_prob

    def transition(
        self,
        state: StateVector,
        action: ActionVector,
        random_event: RandomEventVector,
    ) -> tuple[StateVector, Reward]:
        """Compute next state and reward for a transition.

        Processes one step of the platelet inventory system:
            1. Place replenishment order
            2. Immediately receive the order, where the remaining useful life of the units at arrival is sampled from a multinomial distribution with parameters that may depend on the order quantity
            3. Sample demand from weekday-specific truncated negative binomial distribution
            4. Issue stock using OUFO (Oldest Units First Out) policy
            5. Age remaining stock one period and discard expired units
            6. Reward is negative of total costs:
                - Variable ordering costs (per unit ordered)
                - Fixed ordering costs (when order > 0)
                - Shortage costs (per unit of unmet demand)
                - Wastage costs (per unit that expires)
                - Holding costs (per unit in stock at end of period, including expiring units)
            7. Update weekday to next day of week

        Args:
            state: Current state vector [state_dim]
            action: Action vector [action_dim]
            random_event: Random event vector [event_dim]

        Returns:
            Tuple containing the next state vector [state_dim] and the immediate reward
        """
        demand = random_event[self.random_event_component_lookup["demand"]]
        max_stock_received = random_event[
            self.random_event_component_lookup["stock_received"]
        ]
        opening_stock_after_delivery = (
            jnp.hstack(
                [
                    0,
                    state[self.state_component_lookup["stock"]],
                ]
            )
            + max_stock_received
        )

        # Limit any one element of opening stock to max_order_quantity
        # Assume any units that would take an element over this are
        # not accepted at delivery
        opening_stock_after_delivery = opening_stock_after_delivery.clip(
            0, self.max_order_quantity
        )

        stock_after_issue = self._issue_oufo(opening_stock_after_delivery, demand)

        # Compute variables required to calculate the cost
        order_quantity = action[self.action_component_lookup["order_quantity"]]
        variable_order = order_quantity
        fixed_order = order_quantity > 0
        shortage = jnp.max(
            jnp.array([demand - jnp.sum(opening_stock_after_delivery), 0])
        )
        expiries = stock_after_issue[-1]
        # Note that unlike De Moor scenario, holding costs include units about to expire
        holding = jnp.sum(stock_after_issue)
        closing_stock = stock_after_issue[0 : self.max_useful_life - 1]

        # These components must be in the same order as self.cost_components
        transition_function_reward_output = jnp.array(
            [variable_order, fixed_order, shortage, expiries, holding]
        )

        reward = self._calculate_single_step_reward(
            state, action, transition_function_reward_output
        )

        # Update the weekday
        next_weekday = (state[self.state_component_lookup["weekday"]] + 1) % 7

        next_state = jnp.hstack([next_weekday, closing_stock]).astype(jnp.int32)

        return next_state, reward

    # Supporting functions for __init__()
    # ----------------------------------

    def _useful_life_at_arrival_distribution_valid(
        self,
        useful_life_at_arrival_distribution_c_0: list[float],
        useful_life_at_arrival_distribution_c_1: list[float],
        max_useful_life: int,
    ) -> bool:
        """Check that the useful life at arrival distribution parameters are valid.

        Args:
            useful_life_at_arrival_distribution_c_0: Base logit parameters for useful life at arrival
            useful_life_at_arrival_distribution_c_1: Order quantity multiplier for useful life logits
            max_useful_life: Number of periods before stock expires

        Returns:
            True if parameters are valid, raises AssertionError otherwise
        """
        assert (
            len(useful_life_at_arrival_distribution_c_0) == max_useful_life - 1
        ), "Useful life at arrival distribution params should include an item for c_0 \
            with max_useful_life - 1 parameters"
        assert (
            len(useful_life_at_arrival_distribution_c_1) == max_useful_life - 1
        ), "Useful life at arrival distribution params should include an item for c_1 \
            with max_useful_life - 1 parameters"
        return True

    # Transition function helper methods
    # ----------------------------------

    def _construct_state_component_lookup(self) -> dict[str, int | slice]:
        """Build mapping from state components to indices."""
        return {
            "weekday": 0,  # single index
            "stock": slice(1, self.max_useful_life),  # slice for array
        }

    def _construct_action_component_lookup(self) -> dict[str, int]:
        """Build mapping from action components to indices."""
        return {
            "order_quantity": 0,
        }

    def _construct_random_event_component_lookup(self) -> dict[str, int | slice]:
        """Build mapping from random event components to indices."""
        return {
            "demand": 0,  # single index
            "stock_received": slice(1, self.max_useful_life + 1),  # slice for array
        }

    def _issue_oufo(
        self, opening_stock: Float[Array, "max_useful_life"], demand: int
    ) -> Float[Array, "max_useful_life"]:
        """Issue stock using OUFO (Oldest Units First Out) policy.

        Issues stock starting with oldest items first (right side of vector).
        Uses scan to process each age category in sequence.

        Args:
            opening_stock: Current stock levels by age [max_useful_life]
            demand: Total customer demand to satisfy

        Returns:
            Array of updated stock levels after issuing [max_useful_life]
        """
        # Oldest stock on RHS of vector, so reverse
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
        transition_function_reward_output: Float[Array, "5"],
    ) -> Reward:
        """Calculate reward (negative costs) for one transition step.

        Computes total reward by combining:
            - Variable ordering costs (per unit ordered)
            - Fixed ordering costs (when order > 0)
            - Shortage costs (per unit of unmet demand)
            - Wastage costs (per unit that expires)
            - Holding costs (per unit in stock at end of period, including units about to expire)

        Args:
            state: Current state vector [state_dim]
            action: Action vector [action_dim]
            transition_function_reward_output: Array of cost components [5]

        Returns:
            Negative of total costs for this step
        """
        cost = jnp.dot(transition_function_reward_output, self.cost_components)
        reward = -1 * cost
        return reward

    # Random event probability helper methods
    # ---------------------------------------

    def _get_multinomial_logits(self, action: int) -> Float[Array, "max_useful_life"]:
        """Calculate multinomial logits for useful life at arrival distribution.

        Args:
            action: Order quantity

        Returns:
            Array of logits for each possible useful life [max_useful_life]
        """
        c_0 = self.useful_life_at_arrival_distribution_c_0
        c_1 = self.useful_life_at_arrival_distribution_c_1
        # Assume logit for useful_life=1 is 0, concatenate with logits
        # for other ages using provided coefficients and order size action

        # Parameters are provided in ascending remaining useful life
        # So reverse to match ordering of stock array which is in
        # descending order of remaining useful life so that oldest
        # units are on the RHS
        return jnp.hstack([0, c_0 + (c_1 * action)])[::-1]

    def _calculate_demand_probabilities(
        self, weekday: int
    ) -> Float[Array, "max_demand_plus_one"]:
        """Calculate probabilities for each possible demand value.

        Uses negative binomial distribution with weekday-specific parameters.

        Args:
            weekday: Current weekday (0=Monday to 6=Sunday)

        Returns:
            Array of probabilities for each demand value [max_demand + 1]
        """
        n = self.weekday_demand_negbin_n[weekday]
        p = self.weekday_demand_negbin_p[weekday]

        # NegBin distribution over successes until observe `total_count` failures
        demand_dist = numpyro.distributions.NegativeBinomialProbs(
            total_count=n, probs=(1 - p)
        )
        demand_probs = jnp.exp(demand_dist.log_prob(jnp.arange(0, self.max_demand + 1)))

        # Truncate distribution by adding probability mass for demands > max_demand
        demand_probs = demand_probs.at[self.max_demand].add(1 - jnp.sum(demand_probs))

        return demand_probs

    def _calculate_received_order_probabilities(
        self, action: int, received_order: Float[Array, "max_useful_life"]
    ) -> float:
        """Calculate probabilities for each possible order receipt combination.

        Uses multinomial distribution with logits based on useful life parameters.

        Args:
            action: Order quantity
            received_order: Stock received by age [max_useful_life]

        Returns:
            Probability of this combination of received stock by age
        """
        multinomial_logits = self._get_multinomial_logits(action)
        dist = numpyro.distributions.Multinomial(
            logits=multinomial_logits, total_count=action
        )
        # Only allow combinations that sum to action
        return jnp.where(
            received_order.sum() == action,
            jnp.exp(dist.log_prob(received_order)),
            0,
        )
