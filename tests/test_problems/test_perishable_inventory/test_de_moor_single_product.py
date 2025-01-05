"""Tests for De Moor perishable inventory problem."""

import jax.numpy as jnp
import pytest
import scipy.stats

from mdpax.problems.perishable_inventory.de_moor_single_product import (
    DeMoorSingleProductPerishable,
)

# Policy on this problem using ValueIteration solver compared to
# known results in tests/solvers/test_value_iteration.py


@pytest.mark.parametrize(
    "params,expected_spaces",
    [
        pytest.param(
            {
                "max_useful_life": 2,
                "lead_time": 1,
                "wastage_cost": 7,
                "issue_policy": "lifo",
            },
            {
                "n_states": 121,
                "n_actions": 11,
                "n_random_events": 101,
            },
            id="de_moor/m2/exp1",
        ),
        pytest.param(
            {
                "max_useful_life": 3,
                "lead_time": 2,
                "wastage_cost": 7,
                "issue_policy": "lifo",
            },
            {
                "n_states": 14_641,
                "n_actions": 11,
                "n_random_events": 101,
            },
            id="de_moor/m3/exp5",
        ),
        pytest.param(
            {
                "max_useful_life": 5,
                "lead_time": 2,
                "wastage_cost": 10,
                "issue_policy": "fifo",
            },
            {
                "n_states": 1_771_561,
                "n_actions": 11,
                "n_random_events": 101,
            },
            id="de_moor/m5/exp8",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_space_construction(params, expected_spaces):
    """Test construction of state, action, and random event spaces.

    Verifies that:
    - State space size = (max_order_quantity + 1)^max_useful_life * (max_order_quantity + 1)^(lead_time - 1)
    - Action space size = max_order_quantity + 1
    - Random event space size = max_demand + 1
    """
    problem = DeMoorSingleProductPerishable(**params)

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
    "lead_time,state,action,random_event,expected_next_state,expected_reward",
    [
        (
            1,
            jnp.array([1, 1]),  # [new_stock, old_stock]
            jnp.array([1]),  # order 1 unit
            jnp.array([0]),  # demand 0 units
            jnp.array([1, 1]),  # [received_order, remaining_stock]
            -11.0,  # variable_order = 3, shortage_cost = 0, wastage_cost = 7, holding_cost = 1
        ),
        (
            1,
            jnp.array([1, 1]),  # [new_stock, old_stock]
            jnp.array([1]),  # order 1 unit
            jnp.array([3]),  # demand 3 units
            jnp.array([1, 0]),  # [received_order, remaining_stock]
            -8.0,  # variable_order = 3, shortage_cost = 5, wastage_cost = 0, holding_cost = 0
        ),
        (
            2,
            jnp.array([0, 1, 1]),  # [in_transit, new_stock, old_stock]
            jnp.array([1]),  # order 1 unit
            jnp.array([0]),  # demand 0 units
            jnp.array([1, 0, 1]),  # [new_in_transit, received_order, remaining_stock]
            -11.0,  # variable_order = 3, shortage_cost = 0, wastage_cost = 7, holding_cost = 1
        ),
        (
            2,
            jnp.array([0, 1, 1]),  # [in_transit, new_stock, old_stock]
            jnp.array([5]),  # order 5 units
            jnp.array([3]),  # demand 3 units
            jnp.array([5, 0, 0]),  # [new_in_transit, received_order, remaining_stock]
            -20.0,  # variable_order = 15, shortage_cost = 5, wastage_cost = 0, holding_cost = 0
        ),
    ],
)
def test_transition(
    lead_time, state, action, random_event, expected_next_state, expected_reward
):
    """Test specific transitions have expected outcomes."""
    problem = DeMoorSingleProductPerishable(lead_time=lead_time)
    next_state, reward = problem.transition(state, action, random_event)
    assert jnp.array_equal(next_state, expected_next_state), "Next state doesn't match"
    assert reward == pytest.approx(expected_reward), "Reward doesn't match"


@pytest.mark.parametrize(
    "state,action,random_event,demand_gamma_mean,demand_gamma_cov",
    [
        (
            jnp.array([1, 1]),  # [new_stock, old_stock]
            jnp.array([1]),  # order quantity
            jnp.array([0]),  # no demand
            4.0,  # default mean
            0.5,  # default cov
        ),
        (
            jnp.array([1, 1]),  # [new_stock, old_stock]
            jnp.array([1]),  # order quantity
            jnp.array([4]),  # demand = mean
            4.0,  # default mean
            0.5,  # default cov
        ),
        (
            jnp.array([1, 1]),  # [new_stock, old_stock]
            jnp.array([1]),  # order quantity
            jnp.array([8]),  # high demand
            4.0,  # default mean
            0.5,  # default cov
        ),
        (
            jnp.array([1, 1]),  # [new_stock, old_stock]
            jnp.array([1]),  # order quantity
            jnp.array([100]),  # max demand
            4.0,  # default mean
            0.5,  # default cov
        ),
    ],
)
def test_random_event_probability(
    state, action, random_event, demand_gamma_mean, demand_gamma_cov
):
    """Test random event probabilities match expected values from gamma distribution."""

    problem = DeMoorSingleProductPerishable(
        demand_gamma_mean=demand_gamma_mean, demand_gamma_cov=demand_gamma_cov
    )
    prob = problem.random_event_probability(state, action, random_event)

    # Calculate gamma parameters
    alpha = 1 / (demand_gamma_cov**2)  # shape parameter
    beta = 1 / (demand_gamma_mean * demand_gamma_cov**2)  # rate parameter
    scale = 1 / beta  # scipy uses scale = 1/rate

    # Calculate probability mass for this demand value
    # by integrating between d-0.5 and d+0.5
    d = random_event[0]
    if d == 0:
        expected_prob = scipy.stats.gamma.cdf(0.5, a=alpha, scale=scale)
    elif d == problem.max_demand:
        # For max demand, add all remaining probability mass
        expected_prob = 1 - scipy.stats.gamma.cdf(d - 0.5, a=alpha, scale=scale)
    else:
        expected_prob = scipy.stats.gamma.cdf(
            d + 0.5, a=alpha, scale=scale
        ) - scipy.stats.gamma.cdf(d - 0.5, a=alpha, scale=scale)

    assert prob == pytest.approx(expected_prob, rel=1e-3)


@pytest.mark.parametrize(
    "state,expected_value",
    [
        (
            jnp.array([0, 0, 0]),  # [in_transit, new_stock, old_stock]
            0.0,
        ),
        (
            jnp.array([0, 2, 1]),
            0.0,
        ),
        (
            jnp.array([1, 1, 0]),
            0.0,
        ),
    ],
)
def test_initial_value(state, expected_value):
    """Test initial value estimates for different states - all zero"""
    problem = DeMoorSingleProductPerishable()
    value = problem.initial_value(state)
    assert value == pytest.approx(expected_value, rel=1e-3)


def test_fifo_issuing():
    """Test FIFO issuing policy issues oldest stock first."""
    problem = DeMoorSingleProductPerishable(issue_policy="fifo")
    state = jnp.array([2, 1])  # [new_stock, old_stock]
    action = jnp.array([3])  # Order 3 units
    random_event = jnp.array([2])  # demand 2 units

    next_state, _ = problem.transition(state, action, random_event)

    # Should use old stock (1) first, then 1 from new stock
    assert next_state[1] == 1  # 1 unit of new stock remains, now aged
    assert next_state[0] == 3  # Match order quantity


def test_lifo_issuing():
    """Test LIFO issuing policy issues newest stock first."""
    problem = DeMoorSingleProductPerishable(issue_policy="lifo")
    state = jnp.array([2, 1])  # [new_stock, old_stock]
    action = jnp.array([3])  # Order 3 units
    random_event = jnp.array([2])  # demand 2 units

    next_state, _ = problem.transition(state, action, random_event)
    # Should use new stock (2) first, leaving old stock, which expires
    assert next_state[1] == 0  # 0 units of new stock, now aged
    assert next_state[0] == 3  # Match order quantity
