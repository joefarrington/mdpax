"""Perishable inventory MDP problem from De Moor et al. (2022)."""

import functools
from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpyro
from hydra.conf import dataclass

from mdpax.core.problem import Problem, ProblemConfig


@dataclass
class DeMoorPerishableConfig(ProblemConfig):
    """Configuration for the De Moor Perishable problem."""

    _target_: str = "mdpax.problems.de_moor_perishable.DeMoorPerishable"
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


class DeMoorPerishable(Problem):
    """Class for de_moor_perishable scenario

    Args:
        max_demand: maximum daily demand
        demand_gamma_mean: mean of gamma distribution for demand
        demand_gamma_cov: coefficient of variation of gamma distribution for demand
        max_useful_life: maximum useful life of product, m >= 1
        lead_time: lead time of product, L >= 1
        max_order_quantity: maximum order quantity
        variable_order_cost: cost per unit ordered
        shortage_cost: cost per unit of demand not met
        wastage_cost: cost per unit of product that expires before use
        holding_cost: cost per unit of product in stock at the end of the day
        issue_policy: should be either 'fifo' or 'lifo'
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
        # Paper provides mean, CoV for gamma dist, but numpyro distribution expects
        # alpha (concentration) and beta (rate)
        self.demand_gamma_mean = demand_gamma_mean
        self.demand_gamma_cov = demand_gamma_cov

        (
            self.demand_gamma_alpha,
            self.demand_gamma_beta,
        ) = self._convert_gamma_parameters(
            self.demand_gamma_mean, self.demand_gamma_cov
        )
        self.demand_probabilities = self._calculate_demand_probabilities(
            self.demand_gamma_alpha, self.demand_gamma_beta
        )
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
        else:
            self._issue_stock = self._issue_lifo

        super().__init__()

        self.state_component_lookup = self._construct_state_component_lookup()
        self.event_component_lookup = self._construct_event_component_lookup()

    @property
    def name(self) -> str:
        """Name of the problem."""
        return "de_moor_perishable"

    def _construct_state_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return min and max values for each state dimension.

        State dimensions are:
        - First (L-1) dimensions: in-transit orders [0, max_order_quantity]
        - Next M dimensions: stock at each age [0, max_order_quantity]

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            (mins, maxs) where each is shape [n_dims] and n_dims = L-1 + M
        """
        n_dims = self.max_useful_life + self.lead_time - 1

        # All dimensions bounded by [0, max_order_quantity]
        mins = jnp.zeros(n_dims, dtype=jnp.int32)
        maxs = jnp.full(n_dims, self.max_order_quantity, dtype=jnp.int32)

        return mins, maxs

    def _construct_action_space(self) -> jnp.ndarray:
        """Return array of actions, order quantities from 0 to max_order_quantity."""
        return jnp.arange(0, self.max_order_quantity + 1)

    def action_components(self) -> list[str]:
        """Return list of action component names."""
        return ["order_quantity"]  # only one action component, the order quantity

    def _construct_random_event_space(self) -> jnp.ndarray:
        """Return array of random events, demand between 0 and max_demand."""
        return jnp.arange(0, self.max_demand + 1).reshape(-1, 1)

    @functools.partial(jax.jit, static_argnums=(0,))
    def random_event_probabilities(
        self, state: jnp.ndarray, action: jnp.ndarray
    ) -> float:
        """Compute probability of the random event given state and action.

        Demand is generated from a gamma distribution with mean demand_gamma_mean
        and CoV demand_gamma_cov and is independent of the state and action.
        """
        return self.demand_probabilities

    @functools.partial(jax.jit, static_argnums=(0,))
    def transition(
        self, state: jnp.ndarray, action: jnp.ndarray, random_event: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute next state and reward for forest transition."""
        demand = random_event[self.event_component_lookup["demand"]]

        opening_in_transit = state[self.state_component_lookup["in_transit"]]

        opening_stock = state[self.state_component_lookup["stock"]]

        in_transit = jnp.hstack([action, opening_in_transit])

        stock_after_issue = self._issue_stock(opening_stock, demand)

        # Compute variables required to calculate the cost
        variable_order = action
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

    @functools.partial(jax.jit, static_argnums=(0,))
    def initial_value(self, state: jnp.ndarray) -> float:
        """Initial value estimate based on immediate cut reward."""
        return 0.0

    def get_problem_config(self) -> DeMoorPerishableConfig:
        """Get problem configuration for reconstruction.

        This method should return a ProblemConfig instance containing all parameters
        needed to reconstruct this problem instance. The config will be used during
        checkpoint restoration to recreate the problem.

        Returns:
            Problem configuration
        """
        return DeMoorPerishableConfig(
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

    ################################################################
    ### Supporting functions for self.transition() ###
    ################################################################

    def _construct_state_component_lookup(self) -> dict[str, int]:
        """Build dictionary that maps from named state components to index or slice"""

        return {
            "in_transit": slice(0, self.lead_time - 1),
            "stock": slice(self.lead_time - 1, self.max_useful_life),
        }

    def _construct_event_component_lookup(self) -> dict[str, int]:
        """Return a dictionary that maps name of event component to an index or slice"""
        return {"demand": 0}

    def _issue_fifo(self, opening_stock: chex.Array, demand: int) -> chex.Array:
        """Issue stock using FIFO policy"""
        # Oldest stock on RHS of vector, so reverse
        _, remaining_stock = jax.lax.scan(
            self._issue_one_step, demand, opening_stock, reverse=True
        )
        return remaining_stock

    def _issue_lifo(self, opening_stock: chex.Array, demand: int) -> chex.Array:
        """Issue stock using LIFO policy"""
        # Freshest stock on LHS of vector
        _, remaining_stock = jax.lax.scan(self._issue_one_step, demand, opening_stock)
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
        cost = jnp.dot(transition_function_reward_output, self.cost_components)
        # Multiply by -1 to reflect the fact that they are costs
        reward = -1 * cost
        return reward

    ################################################################
    ### Supporting functions for self.random_event_probability() ###
    ###############################################################

    def _convert_gamma_parameters(self, mean: float, cov: float) -> Tuple[float, float]:
        """Convert mean and coefficient of variation to gamma distribution parameters
        required by numpyro.distributions.Gamma"""
        alpha = 1 / (cov**2)
        beta = 1 / (mean * cov**2)
        return alpha, beta

    def _calculate_demand_probabilities(
        self, gamma_alpha: float, gamma_beta: float
    ) -> chex.Array:
        """Calculate the probability of each demand level (0, max_demand), given the
        gamma distribution parameters"""
        cdf = numpyro.distributions.Gamma(gamma_alpha, gamma_beta).cdf(
            jnp.hstack([0, jnp.arange(0.5, self.max_demand + 1.5)])
        )
        # Want integer demand, so calculate P(d<x+0.5) - P(d<x-0.5),
        # except for 0 demand where use 0 and 0.5
        # This gives us the same results as in Fig 3 of the paper
        demand_probabilities = jnp.diff(cdf)
        # To make number of random outcomes finite, we truncate the distribution
        # Add any probability mass that is truncated back to the last demand level
        demand_probabilities = demand_probabilities.at[-1].add(
            1 - demand_probabilities.sum()
        )
        return demand_probabilities
