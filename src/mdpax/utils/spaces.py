"""Utilities for working with state and action spaces."""

import itertools
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int


def construct_space_from_bounds(
    bounds: Tuple[Int[Array, "dim"], Int[Array, "dim"]]
) -> Int[Array, "n_elements dim"]:
    """Construct a discrete space from lower and upper bounds.

    Creates an array containing all possible integer vectors within the given bounds,
    where each dimension ranges from its lower to upper bound (inclusive).

    When the state space is generated from bounds, the state index can be computed
    using `state_with_dimensions_to_index`.

    Args:
        bounds: Tuple of (mins, maxs) arrays, each of shape [dim], specifying
            the lower and upper bounds (inclusive) for each dimension.

    Returns:
        Array of shape [n_elements, dim] containing all possible integer vectors
        within the bounds, where n_elements is the product of the dimension sizes.

    Example:
        >>> mins = jnp.array([0, 0])
        >>> maxs = jnp.array([1, 2])
        >>> space = construct_space_from_bounds((mins, maxs))
        >>> print(space)
        [[0 0]
         [0 1]
         [0 2]
         [1 0]
         [1 1]
         [1 2]]
    """
    mins, maxs = bounds
    ranges = [np.arange(min_val, max_val + 1) for min_val, max_val in zip(mins, maxs)]
    # More efficient to convert to numpy array first
    space_members = np.array(list(itertools.product(*ranges)), dtype=jnp.int32)
    return jnp.array(space_members)


def space_dimensions_from_bounds(
    bounds: Tuple[Int[Array, "dim"], Int[Array, "dim"]]
) -> Tuple[int, ...]:
    """Calculate the size of each dimension in a space from its bounds.

    Args:
        bounds: Tuple of (mins, maxs) arrays, each of shape [dim], specifying
            the lower and upper bounds (inclusive) for each dimension.

    Returns:
        Tuple of dimension sizes, where each size is maxs[i] - mins[i] + 1.

    Example:
        >>> mins = jnp.array([0, 0])
        >>> maxs = jnp.array([1, 2])
        >>> dims = space_dimensions_from_bounds((mins, maxs))
        >>> print(dims)
        (2, 3)
    """
    mins, maxs = bounds
    return tuple(jnp.array(maxs - mins + 1, dtype=jnp.int32))


def space_with_dimensions_to_index(
    vector: Int[Array, "dim"], space_dimensions: Tuple[int, ...]
) -> int:
    """Convert a vector to a flat index based on dimension sizes.

    Uses row-major (C-style) ordering to compute a unique index for each vector.

    This is designed to be used when the state space is generated from bounds
    using `construct_space_from_bounds` but can also be used with action
    or random event spaces when every integer vector within bounds
    is valid.

    Args:
        vector: Input vector of shape [dim], e.g. member of state space
        space_dimensions: Tuple of dimension sizes, where space_dimensions[i] is
            the size of dimension i.

    Returns:
        Unique integer index for the vector.

    Note:
    The 'clip' mode means any vector will map to a valid index. This is necessary
    for compatibility with JAX's jit compilation but may lead to unexpected results
    if the vector is not within the bounds.
    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ravel_multi_index.html

    Example:
        >>> state = jnp.array([1, 2])
        >>> dimensions = (2, 3)  # 2 possible values in dim 0, 3 in dim 1
        >>> index = space_with_dimensions_to_index(state, dimensions)
        >>> print(index)
        5  # = 1 * 3 + 2 in row-major ordering
    """
    return jnp.ravel_multi_index(tuple(vector), space_dimensions, mode="clip")
