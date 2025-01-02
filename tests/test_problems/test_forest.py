"""Tests for Forest MDP problem."""

import jax.numpy as jnp
import mdptoolbox.example
import pytest

from mdpax.problems.forest import Forest


@pytest.mark.parametrize(
    "params,expected_spaces",
    [
        pytest.param(
            {
                "S": 3,
                "r1": 4.0,
                "r2": 2.0,
                "p": 0.1,
            },
            {
                "n_states": 3,
                "n_actions": 2,
                "n_random_events": 2,
            },
            id="forest/default",
        ),
        pytest.param(
            {
                "S": 5,
                "r1": 10.0,
                "r2": 5.0,
                "p": 0.05,
            },
            {
                "n_states": 5,
                "n_actions": 2,
                "n_random_events": 2,
            },
            id="forest/large",
        ),
    ],
)
def test_space_construction(params, expected_spaces):
    """Test construction of state, action, and random event spaces.

    Verifies that:
    - State space size = S (number of tree ages)
    - Action space size = 2 (wait or cut)
    - Random event space size = 2 (fire or no fire)
    """
    problem = Forest(**params)
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
    "params",
    [
        pytest.param(
            {"S": 3, "r1": 4.0, "r2": 2.0, "p": 0.1},
            id="forest/default",
        ),
        pytest.param(
            {"S": 5, "r1": 10.0, "r2": 5.0, "p": 0.05},
            id="forest/large_rewards",
        ),
        pytest.param(
            {"S": 4, "r1": 2.0, "r2": 8.0, "p": 0.2},
            id="forest/high_risk",
        ),
    ],
)
def test_matrices_match_pymdptoolbox(params):
    """Test that transition and reward matrices match pymdptoolbox implementation.

    This test verifies that our implementation produces identical matrices to
    the reference implementation in pymdptoolbox under different parameter settings.
    """
    # Get matrices from both implementations
    P_orig, R_orig = mdptoolbox.example.forest(**params)
    forest = Forest(**params)
    P_new, R_new = forest.build_transition_and_reward_matrices()

    # Compare matrices (both are float arrays)
    assert P_new == pytest.approx(P_orig, rel=1e-5), "Transition matrices don't match"
    assert R_new == pytest.approx(R_orig, rel=1e-5), "Reward matrices don't match"


@pytest.mark.parametrize(
    "state,action,random_event,expected_next_state,expected_reward",
    [
        pytest.param(
            jnp.array([2]),  # Mature tree
            jnp.array([1]),  # Cut
            jnp.array([0]),  # No fire (doesn't matter when cutting)
            jnp.array([0]),  # Reset to young
            2.0,  # r2 for mature
            id="cut_mature",
        ),
        pytest.param(
            jnp.array([1]),  # Middle age
            jnp.array([0]),  # Wait
            jnp.array([0]),  # No fire
            jnp.array([2]),  # Age increases
            0.0,  # No reward for waiting
            id="wait_middle_no_fire",
        ),
        pytest.param(
            jnp.array([1]),  # Middle age
            jnp.array([0]),  # Wait
            jnp.array([1]),  # Fire
            jnp.array([0]),  # Reset to young
            0.0,  # No reward
            id="wait_middle_fire",
        ),
        pytest.param(
            jnp.array([2]),  # Mature
            jnp.array([0]),  # Wait
            jnp.array([0]),  # No fire
            jnp.array([2]),  # Stay mature
            4.0,  # r1 for waiting in mature
            id="wait_mature_no_fire",
        ),
    ],
)
def test_transition(state, action, random_event, expected_next_state, expected_reward):
    """Test specific transitions have expected outcomes.

    Tests key transitions including:
    - Cutting mature trees
    - Waiting with/without fire
    - Aging process
    - Rewards in different states
    """
    problem = Forest()
    next_state, reward = problem.transition(state, action, random_event)
    assert jnp.array_equal(next_state, expected_next_state), "Next state doesn't match"
    assert reward == pytest.approx(expected_reward), "Reward doesn't match"


@pytest.mark.parametrize(
    "state,action,random_event,expected_prob",
    [
        pytest.param(
            jnp.array([1]),  # Any state
            jnp.array([0]),  # Wait
            jnp.array([0]),  # No fire
            0.9,  # 1-p
            id="wait_no_fire",
        ),
        pytest.param(
            jnp.array([1]),  # Any state
            jnp.array([0]),  # Wait
            jnp.array([1]),  # Fire
            0.1,  # p
            id="wait_fire",
        ),
        pytest.param(
            jnp.array([1]),  # Any state
            jnp.array([1]),  # Cut
            jnp.array([0]),  # No fire
            1.0,  # Certain
            id="cut_no_fire",
        ),
        pytest.param(
            jnp.array([1]),  # Any state
            jnp.array([1]),  # Cut
            jnp.array([1]),  # Fire
            0.0,  # Impossible
            id="cut_fire",
        ),
    ],
)
def test_random_event_probability(state, action, random_event, expected_prob):
    """Test random event probabilities match expected values.

    When waiting:
    - No fire probability is 1-p
    - Fire probability is p
    When cutting:
    - No fire probability is 1
    - Fire probability is 0
    """
    problem = Forest(p=0.1)
    prob = problem.random_event_probability(state, action, random_event)
    assert prob == pytest.approx(expected_prob)


@pytest.mark.parametrize(
    "state,expected_value",
    [
        pytest.param(
            jnp.array([0]),  # Young
            0.0,
            id="young",
        ),
        pytest.param(
            jnp.array([1]),  # Middle
            0.0,
            id="middle",
        ),
        pytest.param(
            jnp.array([2]),  # Mature
            0.0,
            id="mature",
        ),
    ],
)
def test_initial_value(state, expected_value):
    """Test initial value estimates for different states.

    All states should have zero initial value.
    """
    problem = Forest()
    value = problem.initial_value(state)
    assert value == pytest.approx(expected_value)
