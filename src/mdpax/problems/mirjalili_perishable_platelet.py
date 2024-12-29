import itertools
from typing import Dict, List, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from hydra.conf import dataclass

from mdpax.core.problem import Problem, ProblemConfig
from mdpax.utils.spaces import (
    construct_space_from_bounds,
    space_dimensions_from_bounds,
    state_with_dimensions_to_index,
)


@dataclass
class MirjaliliPerishablePlateletConfig(ProblemConfig):
    """Configuration for the Mirjalili Perishable Platelet problem."""

    _target_: str = (
        "mdpax.problems.mirjalili_perishable_platelet.MirjaliliPerishablePlatelet"
    )
    max_demand: int = 20
    # [M, T, W, T, F, S, S]
    weekday_demand_negbin_n: tuple[float] = tuple([3.5, 11.0, 7.2, 11.1, 5.9, 5.5, 2.2])
    weekday_demand_negbin_delta: tuple[float] = tuple(
        [5.7, 6.9, 6.5, 6.2, 5.8, 3.3, 3.4]
    )
    max_useful_life: int = 3
    shelf_life_at_arrival_distribution_c_0: tuple[float] = tuple([1.0, 0.5])
    shelf_life_at_arrival_distribution_c_1: tuple[float] = tuple([0.0, 0.0])
    max_order_quantity: int = 20
    variable_order_cost: float = 0.0
    fixed_order_cost: float = 10.0
    shortage_cost: float = 20.0
    wastage_cost: float = 5.0
    holding_cost: float = 1.0


WEEKDAYS = [
    "Monday",  # idx 0
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",  # idx 6
]


class MirjaliliPerishablePlatelet(Problem):
    """Class to run value iteration for mirjalili_perishable_platelet scenario

    Args:
        max_demand: int,
        weekday_demand_negbin_n: parameter n of the negative binomial distribution,
            one for each weekday in order [M, T, W, T, F, S, S]
        weekday_demand_negbin_delta: parameter delta of the negative binomial
            distribution, one for each weekday in order [M, T, W, T, F, S, S]
        max_useful_life: maximum useful life of product, m >= 1
        shelf_life_at_arrival_distribution_c_0: parameter c_0 used to determine
            parameters of multinomial distribution of useful life on arrival in
            order [2, ..., m]
        shelf_life_at_arrival_distribution_c_1: parameter c_1 used to determine
            parameters of multinomial distribution of useful life on arrival in
            order [2, ..., m]
        max_order_quantity: maximum order quantity
        variable_order_cost: cost per unit ordered
        fixed_order_cost: cost incurred when order > 0
        shortage_cost: cost per unit of demand not met
        wastage_cost: cost per unit of product that expires before use
        holding_cost: cost per unit of product in stock at the end of the day
    """

    def __init__(
        self,
        max_demand: int = 20,
        # [M, T, W, T, F, S, S]
        weekday_demand_negbin_n: List[float] = [3.5, 11.0, 7.2, 11.1, 5.9, 5.5, 2.2],
        weekday_demand_negbin_delta: List[float] = [5.7, 6.9, 6.5, 6.2, 5.8, 3.3, 3.4],
        max_useful_life: int = 3,
        shelf_life_at_arrival_distribution_c_0: List[float] = [1.0, 0.5],
        shelf_life_at_arrival_distribution_c_1: List[float] = [0.0, 0.0],
        max_order_quantity: int = 20,
        variable_order_cost: float = 0.0,
        fixed_order_cost: float = 10.0,
        shortage_cost: float = 20.0,
        wastage_cost: float = 5.0,
        holding_cost: float = 1.0,
    ):

        assert (
            max_useful_life >= 1
        ), "max_useful_life must be greater than or equal to 1"
        self._shelf_life_at_arrival_distribution_valid(
            shelf_life_at_arrival_distribution_c_0,
            shelf_life_at_arrival_distribution_c_1,
            max_useful_life,
        )

        self.max_demand = max_demand

        self.shelf_life_at_arrival_distribution_c_0 = jnp.array(
            shelf_life_at_arrival_distribution_c_0
        )
        self.shelf_life_at_arrival_distribution_c_1 = jnp.array(
            shelf_life_at_arrival_distribution_c_1
        )

        self.weekday_demand_negbin_n = jnp.array(weekday_demand_negbin_n)
        self.weekday_demand_negbin_delta = jnp.array(weekday_demand_negbin_delta)

        self.max_useful_life = max_useful_life
        self.max_order_quantity = max_order_quantity
        self.cost_components = jnp.array(
            [
                variable_order_cost,
                fixed_order_cost,
                shortage_cost,
                wastage_cost,
                holding_cost,
            ]
        )

        super().__init__()

    @property
    def name(self) -> str:
        """Name of the problem."""
        return "mirjalili_perishable_platelet"

    def _setup_before_space_construction(self) -> None:
        """Setup before space construction."""
        # Calculate probability of success, from parameterisation provided in MM thesis
        self.weekday_demand_negbin_p = self.weekday_demand_negbin_n / (
            self.weekday_demand_negbin_delta + self.weekday_demand_negbin_n
        )

        self._state_dimensions = space_dimensions_from_bounds(self._state_bounds)
        self.state_component_lookup = self._construct_state_component_lookup()
        self.event_component_lookup = self._construct_event_component_lookup()

    def _setup_after_space_construction(self) -> None:
        """Setup after space construction."""
        pass

    @property
    def _state_bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return min and max values for each state dimension.

        State dimensions are:
        - First dimension: weekday [0, 6]
        - Next M dimensions: stock at each age [0, max_order_quantity]
        """
        mins = jnp.zeros(self.max_useful_life, dtype=jnp.int32)
        maxs = jnp.hstack(
            [
                jnp.array([6], dtype=jnp.int32),  # weekday
                jnp.full(
                    self.max_useful_life - 1, self.max_order_quantity, dtype=jnp.int32
                ),  # stock
            ]
        )
        return mins, maxs

    def _construct_state_space(self) -> jnp.ndarray:
        """Return array of states, weekday and stock."""
        return construct_space_from_bounds(self._state_bounds)

    def _construct_action_space(self) -> jnp.ndarray:
        """Return array of actions, order quantities from 0 to max_order_quantity."""
        return jnp.arange(0, self.max_order_quantity + 1)

    def _construct_random_event_space(self) -> jnp.ndarray:
        """Return array of random events, demand between 0 and max_demand."""
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

    def state_to_index(self, state: jnp.ndarray) -> int:
        """Convert state vector to index."""
        return state_with_dimensions_to_index(state, self._state_dimensions)

    def random_event_probability(
        self, state: jnp.ndarray, action: jnp.ndarray, random_event: jnp.ndarray
    ) -> float:
        """Compute probability of the random event given state and action.

        Combines demand probabilities (based on weekday) with order receipt
        probabilities (based on action).
        """

        weekday = state[self.state_component_lookup["weekday"]]

        # Get probabilities for demand component
        demand_probs = self._calculate_demand_probabilities(weekday)
        demand_prob = demand_probs[random_event[self.event_component_lookup["demand"]]]

        # Get probabilities for received order component
        received_prob = self._calculate_received_order_probabilities(
            action, random_event[self.event_component_lookup["stock_received"]]
        )

        return demand_prob * received_prob

    def transition(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        random_event: chex.Array,
    ) -> Tuple[chex.Array, float]:
        """A transition in the environment for a given state, action and random event"""
        demand = random_event[self.event_component_lookup["demand"]]
        max_stock_received = random_event[self.event_component_lookup["stock_received"]]
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
        order_quantity = action
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

        # Update the weekday
        next_weekday = (state[self.state_component_lookup["weekday"]] + 1) % 7

        next_state = jnp.hstack([next_weekday, closing_stock]).astype(jnp.int32)

        reward = self._calculate_single_step_reward(
            state, action, transition_function_reward_output
        )

        return next_state, reward

    def initial_value(self, state: jnp.ndarray) -> float:
        """Initial value estimate based on immediate cut reward."""
        return 0.0

    def get_problem_config(self) -> MirjaliliPerishablePlateletConfig:
        """Get problem configuration for reconstruction.

        This method should return a ProblemConfig instance containing all parameters
        needed to reconstruct this problem instance. The config will be used during
        checkpoint restoration to recreate the problem.

        Returns:
            Problem configuration
        """
        return MirjaliliPerishablePlateletConfig(
            max_demand=int(self.max_demand),
            weekday_demand_negbin_n=tuple(
                [float(x) for x in self.weekday_demand_negbin_n]
            ),
            weekday_demand_negbin_delta=tuple(
                [float(x) for x in self.weekday_demand_negbin_delta]
            ),
            max_useful_life=int(self.max_useful_life),
            shelf_life_at_arrival_distribution_c_0=tuple(
                [float(x) for x in self.shelf_life_at_arrival_distribution_c_0]
            ),
            shelf_life_at_arrival_distribution_c_1=tuple(
                [float(x) for x in self.shelf_life_at_arrival_distribution_c_1]
            ),
            max_order_quantity=int(self.max_order_quantity),
            variable_order_cost=float(self.cost_components[0]),
            fixed_order_cost=float(self.cost_components[1]),
            shortage_cost=float(self.cost_components[2]),
            wastage_cost=float(self.cost_components[3]),
            holding_cost=float(self.cost_components[4]),
        )

    ###############################################
    ### Supporting functions for self._init__() ###
    ###############################################

    def _shelf_life_at_arrival_distribution_valid(
        self,
        shelf_life_at_arrival_distribution_c_0: List[float],
        shelf_life_at_arrival_distribution_c_1: List[float],
        max_useful_life: int,
    ) -> bool:
        """Check that the shelf life at arrival distribution parameters are valid"""
        assert (
            len(shelf_life_at_arrival_distribution_c_0) == max_useful_life - 1
        ), "Shelf life at arrival distribution params should include an item for c_0 \
            with max_useful_life - 1 parameters"
        assert (
            len(shelf_life_at_arrival_distribution_c_1) == max_useful_life - 1
        ), "Shelf life at arrival distribution params should include an item for c_1 \
            with max_useful_life - 1 parameters"
        return True

    ################################################################
    ### Supporting functions for self.transition() ###
    ###############################################################

    def _construct_state_component_lookup(self) -> Dict[str, int | slice]:
        """Return indices or slices for state components.

        Returns
        -------
        Dict[str, Union[int, slice]]
            Maps component names to either:
            - int: index for single element access
            - slice: for subarray access
        """
        return {
            "weekday": 0,  # single index
            "stock": slice(1, self.max_useful_life),  # slice for array
        }

    def _construct_event_component_lookup(self) -> Dict[str, int | slice]:
        """Return indices or slices for event components.

        Returns
        -------
        Dict[str, Union[int, slice]]
            Maps component names to either:
            - int: index for single element access
            - slice: for subarray access
        """
        return {
            "demand": 0,  # single index
            "stock_received": slice(1, self.max_useful_life + 1),  # slice for array
        }

    def _issue_oufo(self, opening_stock: chex.Array, demand: int) -> chex.Array:
        """Issue stock using OUFO policy"""
        # Oldest stock on RHS of vector, so reverse
        _, remaining_stock = jax.lax.scan(
            self._issue_one_step, demand, opening_stock, reverse=True
        )
        return remaining_stock

    def _issue_one_step(
        self, remaining_demand: int, stock_element: int
    ) -> Tuple[int, int]:
        """Fill demand with stock of one age, representing one element in the state"""
        remaining_stock = (stock_element - remaining_demand).clip(0)
        remaining_demand = (remaining_demand - stock_element).clip(0)
        return remaining_demand, remaining_stock

    def _calculate_single_step_reward(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        transition_function_reward_output: chex.Array,
    ) -> float:
        """Calculate the single step reward based on the provided state, action and
        output from the transition function"""
        # Minus one to reflect the fact that they are costs
        cost = jnp.dot(transition_function_reward_output, self.cost_components)
        reward = -1 * cost
        return reward

    ###############################################################
    ### Supporting function for self.random_event_probabilities() ###
    ###############################################################

    def _get_multinomial_logits(self, action: int) -> chex.Array:
        """Return multinomial logits for a given order quantity action"""
        c_0 = self.shelf_life_at_arrival_distribution_c_0
        c_1 = self.shelf_life_at_arrival_distribution_c_1
        # Assume logit for useful_life=1 is 0, concatenate with logits
        # for other ages using provided coefficients and order size action

        # Parameters are provided in ascending remaining shelf life
        # So reverse to match ordering of stock array which is in
        # descending order of remaining useful life so that oldest
        # units are on the RHS
        return jnp.hstack([0, c_0 + (c_1 * action)])[::-1]

    def _calculate_demand_probabilities(self, weekday: int) -> jnp.ndarray:
        """Calculate probabilities for each possible demand value.

        Uses negative binomial distribution with weekday-specific parameters.
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
        self, action: int, received_order: jnp.ndarray
    ) -> jnp.ndarray:
        """Calculate probabilities for each possible order receipt combination.

        Uses multinomial distribution with logits based on shelf life parameters.
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
