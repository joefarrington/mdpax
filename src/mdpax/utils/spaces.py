"""Utilities for working with state, action and random event spaces."""

import itertools
from typing import Callable

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


def create_range_space(mins: Array, maxs: Array) -> tuple[Array, Callable]:
    """Create a range-based discrete space and its indexing function.

    Helper function for creating any discrete space (states, actions, or random events)
    that can be represented as ranges in each dimension. Creates both the space
    (all possible vectors) and a function to map vectors to indices.

    The ranges in each dimension are inclusive of both mins and maxs. For example,
    if mins=[0] and maxs=[2], the space will include vectors [0], [1], and [2].

    Args:
        mins: Lower bounds for each dimension [dim], inclusive
        maxs: Upper bounds for each dimension [dim], inclusive

    Returns:
        space: Array of all possible vectors [n_elements, dim]
        index_fn: Function that maps vector to unique index

    Note:
        The 'clip' mode in index_fnmeans any vector will map to a valid index. This is
        necessary for compatibility with JAX's jit compilation but may lead to unexpected
        results if the vector is not within the bounds.
        https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ravel_multi_index.html

    Example:
        >>> # For a state space with 2 dimensions: [0,1] x [0,2] (inclusive)
        >>> state_space, state_to_index = create_range_space(jnp.array([0, 0]), jnp.array([1, 2]))
        >>> print(state_space)  # All 6 combinations including bounds
        [[0 0]
         [0 1]
         [0 2]
         [1 0]
         [1 1]
         [1 2]]
        >>> print(state_to_index(jnp.array([1, 1])))
        4

        >>> # For an action space with 1 dimension: [0,5] (inclusive)
        >>> action_space, action_to_index = create_range_space(jnp.array([0]), jnp.array([5]))
        >>> print(action_space)  # All 6 values from 0 to 5
        [[0]
         [1]
         [2]
         [3]
         [4]
         [5]]
    """
    mins = np.asarray(mins, dtype=np.int32)
    maxs = np.asarray(maxs, dtype=np.int32)
    dimensions = maxs - mins + 1  # +1 because bounds are inclusive
    ranges = [
        np.arange(min_val, max_val + 1)  # +1 to include max_val
        for min_val, max_val in zip(mins, maxs)
    ]
    space = jnp.array(list(itertools.product(*ranges)), dtype=jnp.int32)

    def index_fn(vector: Array) -> int:
        """Map vector to unique index using row-major (C-style) ordering."""
        return jnp.ravel_multi_index(tuple(vector), dimensions, mode="clip")

    return space, index_fn
