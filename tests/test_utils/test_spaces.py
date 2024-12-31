"""Tests for space utilities."""

import jax.numpy as jnp
import pytest

from mdpax.utils.spaces import (
    construct_space_from_bounds,
    space_dimensions_from_bounds,
    space_with_dimensions_to_index,
)


def test_construct_space_from_bounds_basic():
    """Test basic space construction with simple 2D bounds."""
    mins = jnp.array([0, 0])
    maxs = jnp.array([1, 2])
    space = construct_space_from_bounds((mins, maxs))

    # Should create a 6-element space (2 x 3)
    expected = jnp.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    assert jnp.array_equal(space, expected)


def test_space_dimensions_from_bounds():
    """Test calculation of space dimensions from bounds."""
    mins = jnp.array([0, 0, 0])
    maxs = jnp.array([1, 2, 3])
    dimensions = space_dimensions_from_bounds((mins, maxs))

    # Should be (2, 3, 4) - each dimension is max - min + 1
    assert dimensions == (2, 3, 4)
    assert len(dimensions) == 3  # Check dimensionality


@pytest.mark.parametrize(
    "vector,dimensions,expected_index",
    [
        pytest.param(jnp.array([0, 0]), (2, 3), 0, id="first_element"),
        pytest.param(jnp.array([0, 2]), (2, 3), 2, id="last_element_first_row"),
        pytest.param(jnp.array([1, 0]), (2, 3), 3, id="first_element_second_row"),
        pytest.param(jnp.array([1, 2]), (2, 3), 5, id="last_element"),
        pytest.param(jnp.array([0]), (2,), 0, id="single_dimension_first"),
        pytest.param(jnp.array([1]), (2,), 1, id="single_dimension_last"),
        pytest.param(jnp.array([2, 1]), (2, 2), 1, id="wrap_around"),
    ],
)
def test_space_with_dimensions_to_index(vector, dimensions, expected_index):
    """Test conversion of vectors to flat indices."""
    index = space_with_dimensions_to_index(vector, dimensions)
    assert index == expected_index


def test_space_construction_and_indexing():
    """Test the full workflow of constructing a space and indexing into it."""
    mins = jnp.array([0, 0])
    maxs = jnp.array([1, 2])

    # Construct the space
    space = construct_space_from_bounds((mins, maxs))
    dimensions = space_dimensions_from_bounds((mins, maxs))

    # Test that we can recover each vector's position using its index
    for i, vector in enumerate(space):
        index = space_with_dimensions_to_index(vector, dimensions)
        assert index == i
        assert jnp.array_equal(space[index], vector)


def test_space_edge_cases():
    """Test edge cases in space construction and indexing."""
    # Single dimension
    mins = jnp.array([0])
    maxs = jnp.array([1])
    space = construct_space_from_bounds((mins, maxs))
    assert space.shape == (2, 1)

    # Zero-size dimension
    mins = jnp.array([0, 0])
    maxs = jnp.array([0, 1])
    space = construct_space_from_bounds((mins, maxs))
    assert space.shape == (2, 2)

    # Wrap-around indexing
    dimensions = (2, 2)
    vector = jnp.array([2, 1])  # Out of bounds
    index = space_with_dimensions_to_index(vector, dimensions)
    assert index >= 0 and index < 4  # Should wrap to valid index
