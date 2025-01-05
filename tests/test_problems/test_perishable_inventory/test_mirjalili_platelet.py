"""Tests for Mirjalili perishable platelet problem."""

import jax.numpy as jnp
import pytest
import scipy.stats

from mdpax.problems.perishable_inventory.mirjalili_platelet import (
    MirjaliliPlateletPerishable,
)

# Policy on this problem using PeriodicValueIteration solver compared to
# known results in tests/solvers/test_periodic_value_iteration.py


@pytest.mark.parametrize(
    "params,expected_spaces",
    [
        pytest.param(
            {
                "max_useful_life": 3,
            },
            {
                "n_states": 3_087,
                "n_actions": 21,
                "n_random_events": 37_191,
            },
            id="mirjalili/m3/exp1",
        ),
        pytest.param(
            {
                "max_useful_life": 5,
                "useful_life_at_arrival_distribution_c_0": (1.6, 2.6, 2.8, 1.6),
                "useful_life_at_arrival_distribution_c_1": (0, 0, 0, 0),
            },
            {
                "n_states": 1_361_367,
                "n_actions": 21,
                "n_random_events": 1_115_730,
            },
            id="mirjalili/m5/exp1",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_space_construction(params, expected_spaces):
    """Test construction of state, action, and random event spaces.

    Verifies that:
    - State space size = 7 days * (max_order_quantity + 1)^max_useful_life
    - Action space size = max_order_quantity + 1
    - Random event space size = max_demand + 1
    """
    problem = MirjaliliPlateletPerishable(**params)

    assert (
        problem.n_states == expected_spaces["n_states"]
    ), "State space size doesn't match"
    assert (
        problem.n_actions == expected_spaces["n_actions"]
    ), "Action space size doesn't match"
    assert (
        problem.n_random_events == expected_spaces["n_random_events"]
    ), "Random event space size doesn't match"


@pytest.mark.parametrize(
    "state,action,random_event,expected_next_state,expected_reward",
    [
        (
            jnp.array([0, 1, 1]),  # [weekday=Monday, new_stock, old_stock]
            jnp.array([1]),  # order 1 unit
            jnp.array([1, 1, 0, 0]),  # demand 1 unit, unit fresh
            jnp.array([1, 1, 1]),  # [weekday=Tuesday, new_order, remaining_stock]
            -12.0,  # fixed_order = 10, shortage_cost = 0, wastage_cost = 0, holding_cost = 2
        ),
        (
            jnp.array([6, 1, 1]),  # [weekday=Sunday, new_stock, old_stock]
            jnp.array([1]),  # order 1 unit
            jnp.array([4, 1, 0, 0]),  # demand 3 units, unit fresh
            jnp.array([0, 0, 0]),  # [weekday=Monday, new_order, remaining_stock]
            -30.0,  # fixed_order = 10, shortage_cost = 20, wastage_cost = 0, holding_cost = 0
        ),
        (
            jnp.array([2, 1, 1]),  # [weekday=Wednesday, new_stock, old_stock]
            jnp.array([1]),  # order 1 unit
            jnp.array(
                [0, 0, 1, 0]
            ),  # demand 0 units, unit has 2 days remaining useful life
            jnp.array([3, 0, 2]),  # [weekday=Thursday, new_order, remaining_stock]
            -18.0,  # fixed_order = 10, shortage_cost = 0, wastage_cost = 5, holding_cost = 3
        ),
    ],
)
def test_transition(state, action, random_event, expected_next_state, expected_reward):
    """Test specific transitions have expected outcomes."""
    problem = MirjaliliPlateletPerishable()
    next_state, reward = problem.transition(state, action, random_event)
    assert jnp.array_equal(next_state, expected_next_state), "Next state doesn't match"
    assert reward == pytest.approx(expected_reward), "Reward doesn't match"


@pytest.mark.parametrize(
    "state,action,random_event,params",
    [
        # Case 1: Only exogenous uncertainty in useful life
        pytest.param(
            jnp.array([0, 1, 0]),  # Monday, 1 unit in stock
            jnp.array([0]),  # order 1 unit
            jnp.array([5, 0, 0, 0]),  # demand=1, 1 unit with max useful life
            {
                "useful_life_at_arrival_distribution_c_0": (1.0, 0.5),
                "useful_life_at_arrival_distribution_c_1": (0.0, 0.0),
                "max_useful_life": 3,
                "weekday_demand_negbin_n": (
                    3.5,
                    11.0,
                    7.2,
                    11.1,
                    5.9,
                    5.5,
                    2.2,
                ),  # Mon-Sun
                "weekday_demand_negbin_delta": (
                    5.7,
                    6.9,
                    6.5,
                    6.2,
                    5.8,
                    3.3,
                    3.4,
                ),  # Mon-Sun
            },
            id="mirjalili/m3/exp1",
        ),
        # Case 2: Both exogenous and endogenous uncertainty
        pytest.param(
            jnp.array([2, 1, 0]),  # Wednesday, 1 unit in stock
            jnp.array([0]),  # order 2 units
            jnp.array(
                [3, 0, 0, 0]
            ),  # demand=1, 1 unit with useful_life-1, 1 unit with useful_life-2
            {
                "useful_life_at_arrival_distribution_c_0": (1.0, 0.5),
                "useful_life_at_arrival_distribution_c_1": (0.4, 0.8),
                "max_useful_life": 3,
                "weekday_demand_negbin_n": (
                    3.5,
                    11.0,
                    7.2,
                    11.1,
                    5.9,
                    5.5,
                    2.2,
                ),  # Mon-Sun
                "weekday_demand_negbin_delta": (
                    5.7,
                    6.9,
                    6.5,
                    6.2,
                    5.8,
                    3.3,
                    3.4,
                ),  # Mon-Sun
            },
            id="mirjalili/m3/exp2",
        ),
    ],
)
def test_random_event_probability(state, action, random_event, params):
    """Test random event probabilities match expected values.

    The probability of a random event depends on:
    1. Negative binomial demand distribution (weekday-dependent)
       - Parameterized by n (target number of successes) and delta (expected value)
    2. Distribution of remaining useful life at arrival, which can be:
       - Only exogenous (c_1 = 0)
       - Both exogenous and endogenous (c_1 > 0)
       - Uses multinomial distribution for unit ages
    """
    problem = MirjaliliPlateletPerishable(**params)
    prob = problem.random_event_probability(state, action, random_event)

    # Extract components from random event
    demand = random_event[0]
    n_units_by_life = random_event[1:]  # Remaining useful life distribution

    # Get weekday (0=Monday, 6=Sunday)
    weekday = int(state[0])

    # Get negative binomial parameters for this weekday
    n = params["weekday_demand_negbin_n"][weekday]  # number of successes to achieve
    delta = params["weekday_demand_negbin_delta"][weekday]  # expected value

    # Calculate probability of success
    # scipy models number of failures before n successes with prob of success p
    # like the original paper
    p = n / (n + delta)  # probability of success
    prob_demand = scipy.stats.nbinom.pmf(demand, n=n, p=p)  # use p directly

    # Add probability mass for demands > max_demand to max_demand
    if demand == problem.max_demand:
        prob_demand += 1 - scipy.stats.nbinom.cdf(problem.max_demand, n=n, p=p)

    # Calculate probability of useful life distribution
    prob_life = 1.0
    order_quantity = int(action[0])

    if order_quantity > 0:
        # Get base probabilities and order quantity effects
        c_0 = params["useful_life_at_arrival_distribution_c_0"]
        c_1 = params["useful_life_at_arrival_distribution_c_1"]

        # Reverse and append zero coefficient
        # Original: (c_1, c_2, ..., c_m) for 1 to m days
        # Needed: (c_m, c_(m-1), ..., c_1, 0) for 0 to m days
        c_0_full = tuple(reversed(c_0)) + (0.0,)
        c_1_full = tuple(reversed(c_1)) + (0.0,)

        # Calculate probabilities for each life category
        # p = (c_0 + c_1 * order_quantity) / sum(c_0 + c_1 * order_quantity)
        numerators = [
            c_0_full[i] + c_1_full[i] * order_quantity for i in range(len(c_0_full))
        ]
        denominator = sum(numerators)
        probs = [num / denominator for num in numerators]

        # Calculate probability of this distribution using multinomial
        prob_life = scipy.stats.multinomial.pmf(n_units_by_life, order_quantity, probs)

    # Total probability is product of demand and useful life probabilities
    expected_prob = prob_demand * prob_life

    assert prob == pytest.approx(expected_prob, rel=1e-3)


@pytest.mark.parametrize(
    "state,expected_value",
    [
        (
            jnp.array([0, 0, 0]),  # [weekday=Monday, new_stock, old_stock]
            0.0,
        ),
        (
            jnp.array([0, 2, 1]),  # Monday with 3 units total
            0.0,
        ),
        (
            jnp.array([6, 1, 0]),  # Sunday with 1 unit
            0.0,
        ),
    ],
)
def test_initial_value(state, expected_value):
    """Test initial value estimates for different states"""
    problem = MirjaliliPlateletPerishable()
    value = problem.initial_value(state)
    assert value == pytest.approx(expected_value, rel=1e-3)


def test_weekday_dependent_demand():
    """Test demand probabilities vary by weekday."""
    problem = MirjaliliPlateletPerishable(max_demand=5)

    # Compare Monday vs Sunday probabilities for same state/action
    state_monday = jnp.array([0, 1, 0])  # Monday, 1 new unit
    state_sunday = jnp.array([6, 1, 0])  # Sunday, 1 new unit
    action = jnp.array([0])  # no order
    random_event = jnp.array([1, 0, 0, 0])  # demand 1 unit, no units to receive

    prob_monday = problem.random_event_probability(state_monday, action, random_event)
    prob_sunday = problem.random_event_probability(state_sunday, action, random_event)

    # Different weekdays should have different demand probabilities
    assert prob_monday != prob_sunday
