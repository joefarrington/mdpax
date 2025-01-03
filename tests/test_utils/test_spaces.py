"""Tests for space utilities."""

import jax.numpy as jnp

from mdpax.utils.spaces import create_range_space


def test_create_range_space_basic():
    """Test basic space construction and indexing with simple 2D bounds."""
    mins = jnp.array([0, 0])
    maxs = jnp.array([1, 2])
    space, index_fn = create_range_space(mins, maxs)

    # Should create a 6-element space (2 x 3)
    expected = jnp.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    assert jnp.array_equal(space, expected)

    # Test indexing of key positions
    assert index_fn(jnp.array([0, 0])) == 0  # first element
    assert index_fn(jnp.array([0, 2])) == 2  # last element first row
    assert index_fn(jnp.array([1, 0])) == 3  # first element second row
    assert index_fn(jnp.array([1, 2])) == 5  # last element


def test_space_construction_and_indexing():
    """Test the full workflow of constructing a space and indexing into it."""
    mins = jnp.array([0, 0])
    maxs = jnp.array([1, 2])

    # Construct the space
    space, index_fn = create_range_space(mins, maxs)

    # Test that we can recover each vector's position using its index
    for i, vector in enumerate(space):
        index = index_fn(vector)
        assert index == i
        assert jnp.array_equal(space[index], vector)


def test_space_edge_cases():
    """Test edge cases in space construction and indexing."""
    # Single dimension
    mins = jnp.array([0])
    maxs = jnp.array([1])
    space, _ = create_range_space(mins, maxs)
    assert space.shape == (2, 1)

    # Zero-size dimension
    mins = jnp.array([0, 0])
    maxs = jnp.array([0, 1])
    space, _ = create_range_space(mins, maxs)
    assert space.shape == (2, 2)

    # Higher dimensions
    mins = jnp.array([0, 0, 0])
    maxs = jnp.array([1, 1, 1])
    space, index_fn = create_range_space(mins, maxs)
    assert space.shape == (8, 3)  # 2^3 combinations
    assert index_fn(jnp.array([1, 1, 1])) == 7  # last element
