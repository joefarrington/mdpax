"""Tests for Hendrix perishable substitution problem."""

import jax.numpy as jnp
import pytest
import scipy.stats

from mdpax.problems.perishable_inventory.hendrix_two_product import (
    HendrixTwoProductPerishable,
)

# Policy on this problem using RelativeValueIteration solver compared to
# known results in tests/solvers/test_relative_value_iteration.py


@pytest.mark.parametrize(
    "params,expected_spaces",
    [
        pytest.param(
            {
                "max_useful_life": 2,
                "demand_poisson_mean_a": 5.0,
                "demand_poisson_mean_b": 5.0,
                "max_order_quantity_a": 10,
                "max_order_quantity_b": 10,
            },
            {
                "n_states": 14_641,
                "n_actions": 121,
                "n_random_events": 441,
            },
            id="hendrix/m2/exp1",
        ),
        pytest.param(
            {
                "max_useful_life": 3,
                "demand_poisson_mean_a": 5.0,
                "demand_poisson_mean_b": 5.0,
                "max_order_quantity_a": 15,
                "max_order_quantity_b": 15,
            },
            {
                "n_states": 16_777_216,
                "n_actions": 256,
                "n_random_events": 2_116,
            },
            id="hendrix/m3/exp1",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_space_construction(params, expected_spaces):
    """Test construction of state, action, and random event spaces.

    Verifies that:
    - State space size = (max_order_quantity_a + 1)^max_useful_life * (max_order_quantity_b + 1)^max_useful_life
    - Action space size = (max_order_quantity_a + 1) * (max_order_quantity_b + 1)
    - Random event space size = (max_stock_a + 1) * (max_stock_b + 1)
      where max_stock_x = max_order_quantity_x * max_useful_life
    """
    problem = HendrixTwoProductPerishable(**params)

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
            jnp.array(
                [1, 0, 1, 0]
            ),  # [stock_a_new, stock_a_old, stock_b_new, stock_b_old]
            jnp.array([1, 1]),  # order 1 unit of each
            jnp.array([1, 1]),  # issue 1 unit of each
            jnp.array([1, 0, 1, 0]),  # new orders arrive, old stock used
            1.0,  # revenue = 2, variable_order_cost = 1
        ),
    ],
)
def test_transition(state, action, random_event, expected_next_state, expected_reward):
    """Test specific transitions have expected outcomes."""
    problem = HendrixTwoProductPerishable()
    next_state, reward = problem.transition(state, action, random_event)
    assert jnp.array_equal(next_state, expected_next_state), "Next state doesn't match"
    assert reward == pytest.approx(expected_reward), "Reward doesn't match"


@pytest.mark.parametrize(
    "state,action,random_event,poisson_mean_a,poisson_mean_b,substitution_prob",
    [
        (
            jnp.array(
                [5, 0, 5, 0]
            ),  # [stock_a_new, stock_a_old, stock_b_new, stock_b_old]
            jnp.array([5, 5]),  # order 5 units of each
            jnp.array([4, 4]),  # issue 4 unit of each, both below stock
            5.0,  # mean demand for A
            5.0,  # mean demand for B
            0.5,  # substitution probability
        ),
        (
            jnp.array(
                [5, 0, 5, 0]
            ),  # [stock_a_new, stock_a_old, stock_b_new, stock_b_old]
            jnp.array([5, 5]),  # order 5 units of each
            jnp.array([5, 3]),  # issue all stock of A, less than stock of B
            5.0,  # mean demand for A
            5.0,  # mean demand for B
            0.5,  # substitution probability
        ),
        (
            jnp.array(
                [5, 0, 5, 0]
            ),  # [stock_a_new, stock_a_old, stock_b_new, stock_b_old]
            jnp.array([5, 5]),  # order 5 units of each
            jnp.array([3, 5]),  # issue less than stock of A, all stock of B
            5.0,  # mean demand for A
            5.0,  # mean demand for B
            0.5,  # substitution probability
        ),
        (
            jnp.array(
                [5, 0, 5, 0]
            ),  # [stock_a_new, stock_a_old, stock_b_new, stock_b_old]
            jnp.array([5, 5]),  # order 5 units of each
            jnp.array([5, 5]),  # issue all stock of both products
            5.0,  # mean demand for A
            5.0,  # mean demand for B
            0.5,  # substitution probability
        ),
    ],
)
def test_random_event_probability(
    state, action, random_event, poisson_mean_a, poisson_mean_b, substitution_prob
):
    """Test random event probabilities match expected values.

    The probability of issuing units depends on:
    1. Poisson demand for each product
    2. Possible substitution from A to B when B's demand exceeds stock
    3. Available stock levels

    Four cases are tested:
    1. Both products issued less than stock
    2. Product A issued equal to stock, B less than stock
    3. Product A issued less than stock, B equal to stock
    4. Both products issued equal to stock
    """
    problem = HendrixTwoProductPerishable(
        demand_poisson_mean_a=poisson_mean_a,
        demand_poisson_mean_b=poisson_mean_b,
        substitution_probability=substitution_prob,
    )
    prob = problem.random_event_probability(state, action, random_event)

    # Get total stock of each product
    stock_a = jnp.sum(state[0:2])
    stock_b = jnp.sum(state[2:4])
    issued_a = random_event[0]
    issued_b = random_event[1]

    # Calculate probability based on which case we're in
    if issued_a < stock_a and issued_b < stock_b:
        # Case 1: Both less than stock - simple Poisson probabilities
        prob_a = scipy.stats.poisson.pmf(issued_a, poisson_mean_a)
        prob_b = scipy.stats.poisson.pmf(issued_b, poisson_mean_b)
        expected_prob = prob_a * prob_b

    elif issued_a == stock_a and issued_b < stock_b:
        # Case 2: A equals stock (demand ≥ stock), B less than stock
        prob_a = 1 - scipy.stats.poisson.cdf(stock_a - 1, poisson_mean_a)
        prob_b = scipy.stats.poisson.pmf(issued_b, poisson_mean_b)
        expected_prob = prob_a * prob_b

    elif issued_a < stock_a and issued_b == stock_b:
        # Case 3: A less than stock, B equals stock (with possible substitution)
        expected_prob = 0.0

        # Sum over possible direct demands for A up to issued amount
        for d_a in range(issued_a + 1):
            prob_d_a = scipy.stats.poisson.pmf(d_a, poisson_mean_a)

            # Need substitution to make up the difference
            u_needed = issued_a - d_a
            if u_needed < 0:
                continue

            prob_b_and_subst = 0.0

            for d_b in range(stock_b, problem.max_demand):
                excess_b = d_b - stock_b

                if excess_b < u_needed:
                    continue

                prob_d_b = scipy.stats.poisson.pmf(d_b, poisson_mean_b)
                prob_subst = scipy.stats.binom.pmf(
                    u_needed, excess_b, substitution_prob
                )
                prob_b_and_subst += prob_d_b * prob_subst

            expected_prob += prob_d_a * prob_b_and_subst

    else:  # issued_a == stock_a and issued_b == stock_b
        # Case 4: Both equal to stock
        expected_prob = 0.0

        # First handle case where direct A demand ≥ stock_a
        prob_a_excess = 1 - scipy.stats.poisson.cdf(stock_a - 1, poisson_mean_a)
        prob_b_excess = 1 - scipy.stats.poisson.cdf(stock_b - 1, poisson_mean_b)
        expected_prob += prob_a_excess * prob_b_excess

        # Then handle case where we need substitution to reach stock_a
        for d_a in range(
            stock_a
        ):  # Note: not including stock_a as that's handled above
            prob_d_a = scipy.stats.poisson.pmf(d_a, poisson_mean_a)

            # Need substitution to reach stock_a
            u_needed = stock_a - d_a

            # Start from stock_b (minimum demand needed)
            prob_b_and_subst = 0.0

            for d_b in range(stock_b, problem.max_demand):
                excess_b = d_b - stock_b

                if excess_b < u_needed:
                    continue

                prob_d_b = scipy.stats.poisson.pmf(d_b, poisson_mean_b)
                # Need at least u_needed substitutions
                prob_subst = 1 - scipy.stats.binom.cdf(
                    u_needed - 1, excess_b, substitution_prob
                )
                prob_b_and_subst += prob_d_b * prob_subst

            expected_prob += prob_d_a * prob_b_and_subst

    assert prob == pytest.approx(expected_prob, rel=1e-3)


@pytest.mark.parametrize(
    "state,poisson_mean_a,poisson_mean_b,substitution_prob,sales_price_a,sales_price_b",
    [
        (
            jnp.array(
                [0, 0, 0, 0]
            ),  # [stock_a_new, stock_a_old, stock_b_new, stock_b_old]
            1.0,  # mean demand for A
            1.0,  # mean demand for B
            0.5,  # substitution probability
            1.0,  # price of A
            1.0,  # price of B
        ),
        (
            jnp.array([1, 0, 1, 0]),  # One new unit of each product
            1.0,  # mean demand for A
            1.0,  # mean demand for B
            0.5,  # substitution probability
            1.0,  # price of A
            1.0,  # price of B
        ),
        (
            jnp.array([2, 1, 0, 0]),  # Three units of A, no B
            1.0,  # mean demand for A
            1.0,  # mean demand for B
            0.5,  # substitution probability
            1.0,  # price of A
            1.0,  # price of B
        ),
    ],
)
def test_initial_value(
    state,
    poisson_mean_a,
    poisson_mean_b,
    substitution_prob,
    sales_price_a,
    sales_price_b,
):
    """Test initial value estimates for different states.

    Initial value is based on one-step ahead expected sales revenue:
    - Regular sales of each product at their respective prices
    - Possible substitution of A for B when B is out of stock

    The expected revenue is calculated by summing over all possible combinations of:
    1. Demand for A (d_a)
    2. Demand for B (d_b)
    3. Number of B customers willing to substitute (u)

    For each combination:
    - Units of A sold = min(stock_a, d_a + u)
    - Units of B sold = min(stock_b, d_b)
    """
    problem = HendrixTwoProductPerishable(
        sales_price_a=sales_price_a,
        sales_price_b=sales_price_b,
        substitution_probability=substitution_prob,
        demand_poisson_mean_a=poisson_mean_a,
        demand_poisson_mean_b=poisson_mean_b,
    )
    value = problem.initial_value(state)

    # Calculate expected value manually
    stock_a = jnp.sum(state[0:2])
    stock_b = jnp.sum(state[2:4])
    max_demand = problem.max_useful_life * (
        max(problem.max_order_quantity_a, problem.max_order_quantity_b) + 2
    )

    expected_value = 0.0

    # Sum over all possible combinations
    for d_a in range(max_demand + 1):
        prob_a = scipy.stats.poisson.pmf(d_a, poisson_mean_a)

        for d_b in range(max_demand + 1):
            prob_b = scipy.stats.poisson.pmf(d_b, poisson_mean_b)

            # Calculate B sales first
            sales_b = min(d_b, stock_b)

            # If B demand exceeds stock, calculate substitution
            excess_b = max(0, d_b - stock_b)

            # For each possible number of customers willing to substitute
            for u in range(excess_b + 1):
                # Probability of exactly u customers willing to substitute
                prob_u = scipy.stats.binom.pmf(u, excess_b, substitution_prob)

                # Total demand for A is direct demand plus substitution
                total_a_demand = d_a + u
                sales_a = min(total_a_demand, stock_a)

                # Calculate revenue for this combination
                revenue = sales_a * sales_price_a + sales_b * sales_price_b

                # Multiply by probability of this combination occurring
                expected_value += revenue * prob_a * prob_b * prob_u

    assert value == pytest.approx(expected_value, rel=1e-3)
