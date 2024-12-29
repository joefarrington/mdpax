"""Tests for the Forest MDP problem."""

import jax
import jax.numpy as jnp
import mdptoolbox.example
import numpy as np
import pytest

from mdpax.problems.forest import Forest


@pytest.mark.parametrize(
    "params",
    [
        {"S": 3, "r1": 4.0, "r2": 2.0, "p": 0.1},  # Default
        {"S": 5, "r1": 10.0, "r2": 5.0, "p": 0.05},  # Different size and rewards
        {"S": 4, "r1": 2.0, "r2": 8.0, "p": 0.2},  # Higher fire risk
    ],
)
def test_forest_match_mdptoolbox(params):
    """Test that matrices match for different parameter settings."""
    # Get matrices from both implementations
    P_orig, R_orig = mdptoolbox.example.forest(**params)

    forest = Forest(**params)
    P_new, R_new = forest.build_matrices()

    # Convert to numpy
    P_new = np.array(P_new)
    R_new = np.array(R_new)

    # Compare
    np.testing.assert_allclose(
        P_orig, P_new, rtol=1e-5, err_msg="Transition matrices don't match"
    )
    np.testing.assert_allclose(
        R_orig, R_new, rtol=1e-5, err_msg="Reward matrices don't match"
    )


def test_forest_properties():
    """Test basic properties of the forest implementation."""
    forest = Forest()

    # Test dimensions
    assert forest.n_states == forest.S
    assert forest.n_actions == 2  # wait, cut
    assert forest.n_random_events == 2  # no fire, fire

    # Test state indexing
    for i in range(forest.S):
        assert forest.state_to_index(jnp.array([i])) == i

    # Test initial values
    initial_values = jax.vmap(forest.initial_value, in_axes=0)(forest.state_space)
    assert initial_values.shape == (forest.S,)
    assert np.all(initial_values == 0)


def test_forest_transitions():
    """Test specific transitions and rewards."""
    forest = Forest(S=3, r1=4.0, r2=2.0, p=0.1)

    # Test cutting a mature tree
    next_state, reward = forest.transition(
        state=jnp.array([2]),  # Mature tree
        action=forest.action_space[1],  # Cut
        random_event=forest.random_event_space[0],  # No fire (shouldn't matter)
    )
    assert reward == 2.0  # r2 for mature
    assert next_state == 0  # Reset to young

    # Test waiting with no fire
    next_state, reward = forest.transition(
        state=jnp.array([1]),  # Middle age
        action=forest.action_space[0],  # Wait
        random_event=forest.random_event_space[0],  # No fire
    )
    assert reward == 0.0  # No reward for waiting
    assert next_state == 2  # Age increases


def test_forest_probabilities():
    """Test random_event probabilities."""
    forest = Forest(S=3, p=0.1)

    # Test waiting action
    probs = jax.vmap(
        forest.random_event_probability,
        in_axes=(None, None, 0),
    )(
        jnp.array([1]),  # Fixed state
        jnp.array([0]),  # Wait action
        forest.random_event_space,  # Vary random events
    )
    assert np.allclose(probs, [0.9, 0.1])

    # Test cutting action
    probs = jax.vmap(
        forest.random_event_probability,
        in_axes=(None, None, 0),
    )(
        jnp.array([1]),  # Fixed state
        jnp.array([1]),  # Cut action
        forest.random_event_space,  # Vary random events
    )
    assert np.allclose(probs, [1.0, 0.0])
