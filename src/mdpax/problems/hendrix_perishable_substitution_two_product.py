"""Perishable inventory MDP problem from Hendrix et al. (2019)."""

import itertools
from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
from hydra.conf import dataclass

from mdpax.core.problem import Problem, ProblemConfig


@dataclass
class HendrixPerishableSubstitutionTwoProductConfig(ProblemConfig):
    """Configuration for the Hendrix Perishable Substitution Two Product problem."""

    _target_: str = (
        "mdpax.problems.hendrix_perishable_substitution_two_product.HendrixPerishableSubstitutionTwoProduct"
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


class HendrixPerishableSubstitutionTwoProduct(Problem):
    """Class for hendrix_perishable_substitution_two_product scenario

    Args:
        max_useful_life: maximum useful life of product, m >= 1
        demand_poission_mean_a: mean of Poisson distribution of demand for product A
        demand_poission_mean_b: mean of Poisson distribution of demand for product B
        substituion_probability: probability that excess demand for product B can be
            satisfied by product A
        variable_order_cost_a: cost per unit of product A ordered
        variable_order_cost_b: cost per unit of product B ordered
        sales_price_a: revenue per unit of product A issued to meet demand
        sales_price_b: revenue per unit of product B issued to meet demand
        max_order_quantity_a: maximum order quantity for product A
        max_order_quantity_b: maximum order quantity for product B
    """

    def __init__(
        self,
        max_useful_life: int = 2,
        demand_poisson_mean_a: float = 5.0,
        demand_poisson_mean_b: float = 5.0,
        substitution_probability: float = 0.5,
        variable_order_cost_a: float = 0.5,
        variable_order_cost_b: float = 0.5,
        sales_price_a: float = 1.0,
        sales_price_b: float = 1.0,
        max_order_quantity_a: int = 10,
        max_order_quantity_b: int = 10,
    ):

        assert (
            max_useful_life >= 1
        ), "max_useful_life must be greater than or equal to 1"

        self.max_useful_life = max_useful_life
        self.demand_poisson_mean_a = demand_poisson_mean_a
        self.demand_poisson_mean_b = demand_poisson_mean_b
        self.substitution_probability = substitution_probability
        self.variable_order_cost_a = variable_order_cost_a
        self.variable_order_cost_b = variable_order_cost_b
        self.sales_price_a = sales_price_a
        self.sales_price_b = sales_price_b
        self.variable_order_costs = jnp.array(
            [self.variable_order_cost_a, self.variable_order_cost_b]
        )
        self.sales_prices = jnp.array([self.sales_price_a, self.sales_price_b])
        self.max_order_quantity_a = max_order_quantity_a
        self.max_order_quantity_b = max_order_quantity_b

        self.max_stock_a = self.max_order_quantity_a * self.max_useful_life
        self.max_stock_b = self.max_order_quantity_b * self.max_useful_life
        self.max_demand = self.max_useful_life * (
            max(self.max_order_quantity_a, self.max_order_quantity_b) + 2
        )
        super().__init__()
        self.state_component_lookup = self._construct_state_component_lookup()
        self.event_component_lookup = self._construct_event_component_lookup()
        # Precompute conditional probabilities for speed
        self.pu = self._calculate_pu()
        self.pz = self._calculate_pz()

    @property
    def name(self) -> str:
        """Name of the problem."""
        return "hendrix_perishable_substitution_two_product"

    def _construct_state_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return min and max values for each state dimension.

        State dimensions are:
        - First self.max_useful_life dimensions: stock at each age
            [0, max_order_quantity_a + 1]
        - Next self.max_useful_life dimensions: stock at each age
            [0, max_order_quantity_b + 1]

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            (mins, maxs) where each is shape [n_dims] and
            n_dims = 2 * self.max_useful_life
        """
        mins = jnp.zeros(2 * self.max_useful_life, dtype=jnp.int32)
        maxs = jnp.hstack(
            [
                jnp.full(
                    self.max_useful_life, self.max_order_quantity_a, dtype=jnp.int32
                ),
                jnp.full(
                    self.max_useful_life, self.max_order_quantity_b, dtype=jnp.int32
                ),
            ]
        )

        return mins, maxs

    def _construct_action_space(self) -> jnp.ndarray:
        """Return array of actions, pairs of order quantities for products A and B."""
        return jnp.array(
            list(
                itertools.product(
                    range(0, self.max_order_quantity_a + 1),
                    range(0, self.max_order_quantity_b + 1),
                )
            )
        )

    def action_components(self) -> list[str]:
        """Return list of action component names."""
        return ["order_quantity_a", "order_quantity_b"]

    def _construct_random_event_space(self) -> jnp.ndarray:
        """Return array of random events, number of units issued of product A and B."""
        return jnp.array(
            list(
                itertools.product(
                    range(0, self.max_stock_a + 1), range(0, self.max_stock_b + 1)
                )
            )
        )

    def random_event_probabilities(
        self, state: jnp.ndarray, action: jnp.ndarray
    ) -> float:
        """Returns an array of the probabilities of each possible random outcome
        for the provided state-action pair"""
        stock_a = jnp.sum(
            jax.lax.dynamic_slice(
                state,
                (self.state_component_lookup["stock_a"][0],),
                (self.state_component_lookup["stock_a"][1],),
            )
        )
        stock_b = jnp.sum(
            jax.lax.dynamic_slice(
                state,
                (self.state_component_lookup["stock_b"][0],),
                (self.state_component_lookup["stock_b"][1],),
            )
        )
        # Issued a less than stock of a, issued b less than stock of b
        probs_1 = self._get_probs_ia_lt_stock_a_ib_lt_stock_b(stock_a, stock_b)
        # Issued a equal to stock of a, issued b less than stock of b
        probs_2 = self._get_probs_ia_eq_stock_a_ib_lt_stock_b(stock_a, stock_b)
        # Issued a less than stock of a, issued b equal to stock of b
        probs_3 = self._get_probs_ia_lt_stock_a_ib_eq_stock_b(stock_a, stock_b)
        # Issued a equal to stock of a, issued b equal to stock of b
        probs_4 = self._get_probs_ia_eq_stock_a_ib_eq_stock_b(stock_a, stock_b)

        return (probs_1 + probs_2 + probs_3 + probs_4).reshape(-1)

    def transition(
        self,
        state: chex.Array,
        action: Union[int, chex.Array],
        random_event: chex.Array,
    ) -> Tuple[chex.Array, float]:
        """Returns the next state and single-step reward for the provided
        state, action and random event"""
        opening_stock_a = jax.lax.dynamic_slice(
            state,
            (self.state_component_lookup["stock_a"][0],),
            (self.state_component_lookup["stock_a"][1],),
        )
        opening_stock_b = jax.lax.dynamic_slice(
            state,
            (self.state_component_lookup["stock_b"][0],),
            (self.state_component_lookup["stock_b"][1],),
        )

        issued_a = random_event[self.event_component_lookup["issued_a"]]
        issued_b = random_event[self.event_component_lookup["issued_b"]]

        stock_after_issue_a = self._issue_fifo(opening_stock_a, issued_a)
        stock_after_issue_b = self._issue_fifo(opening_stock_b, issued_b)

        # Age stock one day and receive the order from the morning
        closing_stock_a = jnp.hstack(
            [action[0], stock_after_issue_a[0 : self.max_useful_life - 1]]
        )
        closing_stock_b = jnp.hstack(
            [action[1], stock_after_issue_b[0 : self.max_useful_life - 1]]
        )

        next_state = jnp.concatenate([closing_stock_a, closing_stock_b], axis=-1)

        # Pass through the random outcome (units issued)
        single_step_reward = self._calculate_single_step_reward(
            state, action, random_event
        )
        return (
            next_state,
            single_step_reward,
        )

    def initial_value(self, state: jnp.ndarray) -> float:
        """Initial value estimate based on one-step ahead expected sales revenue"""
        return self._calculate_expected_sales_revenue(state)

    def get_problem_config(self) -> HendrixPerishableSubstitutionTwoProductConfig:
        """Get problem configuration for reconstruction.

        This method should return a ProblemConfig instance containing all parameters
        needed to reconstruct this problem instance. The config will be used during
        checkpoint restoration to recreate the problem.

        Returns:
            Problem configuration
        """
        return HendrixPerishableSubstitutionTwoProductConfig(
            max_useful_life=self.max_useful_life,
            demand_poisson_mean_a=self.demand_poisson_mean_a,
            demand_poisson_mean_b=self.demand_poisson_mean_b,
            substitution_probability=self.substitution_probability,
            variable_order_cost_a=self.variable_order_cost_a,
            variable_order_cost_b=self.variable_order_cost_b,
            sales_price_a=self.sales_price_a,
            sales_price_b=self.sales_price_b,
        )

    ##################################################
    ### Supporting functions for self.transition() ###
    ##################################################

    def _construct_state_component_lookup(self) -> dict[str, tuple[int, int]]:
        """Returns a dictionary mapping component names to (start, length)
        tuples for slicing."""
        m = self.max_useful_life
        return {
            "stock_a": (0, m),  # (start, length) for dynamic_slice
            "stock_b": (m, m),  # (start, length) for dynamic_slice
        }

    def _construct_event_component_lookup(self) -> dict[str, int]:
        """Returns a dictionary that maps descriptive names of the components of a event
        to indices of the elements in the event array"""
        event_component_lookup = {}
        event_component_lookup["issued_a"] = 0
        event_component_lookup["issued_b"] = 1
        return event_component_lookup

    def _issue_fifo(self, opening_stock: chex.Array, demand: int) -> chex.Array:
        """Issue stock using FIFO policy"""
        # Oldest stock on RHS of vector, so reverse
        _, remaining_stock = jax.lax.scan(
            self._issue_one_step, demand, opening_stock, reverse=True
        )
        return remaining_stock

    def _issue_one_step(
        self, remaining_demand: chex.Array, stock_element: int
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
        cost = jnp.dot(action, self.variable_order_costs)
        revenue = jnp.dot(transition_function_reward_output, self.sales_prices)
        return revenue - cost

    ##################################################################
    ### Supporting functions for self.random_event_probabilities() ###
    #################################################################

    def _calculate_pu(self) -> np.ndarray:
        """Returns an array of the conditional probabilities of substitution demand
        given that demand for b exceeds stock of b. pu[u,y] is Prob(u|y), conditional
        probability of u substitution demand given y units of b in stock"""

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

        return pu

    #
    # TODO: Could try to rewrite for speed, but only runs once
    def _calculate_pz(self) -> np.ndarray:
        """Returns an array of the conditional probabilities of total demand
        for a given that demand for b is at least equal to total demand for b.
        pz[z,y] is Prob(z|y), conditional probability of z demand from product a given
        demand for product b is at least equal to y, number of units in stock"""
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
        return pz

    def _get_probs_ia_lt_stock_a_ib_lt_stock_b(
        self, stock_a: chex.Array, stock_b: chex.Array
    ) -> chex.Array:
        """Returns probabilities for issued quantities of a and b given that
        issued a < stock a, issued_b < stock_b"""
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
        """Returns probabilities for issued quantities of a and b given that
        issued a = stock a, issued_b < stock_b"""
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
        """Returns probabilities for issued quantities of a and b given that
        issued a < stock a, issued_b = stock_b"""
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
        """Returns probabilities for issued quantities of a and b given that
        issued a = stock a, issued_b = stock_b"""
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

    ##################################################################
    ### Support functions for self.initial_value() ###
    ##################################################################

    def _calculate_sales_revenue_for_possible_random_outcomes(self) -> chex.Array:
        """Calculate the sales revenue for each possible random outcome of demand"""
        return (self.random_event_space.dot(self.sales_prices)).reshape(-1)

    def _calculate_expected_sales_revenue(self, state: chex.Array) -> float:
        """Calculate the expected sales revenue for a given state"""
        issued_probabilities = self.random_event_probabilities(state, 0)
        expected_sales_revenue = issued_probabilities.dot(
            self._calculate_sales_revenue_for_possible_random_outcomes()
        )
        return expected_sales_revenue
