import itertools

import jax.numpy as jnp
import numpy as np


def construct_space_from_bounds(bounds: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """Construct state space from bounds."""
    mins, maxs = bounds
    ranges = [np.arange(min_val, max_val + 1) for min_val, max_val in zip(mins, maxs)]
    # More efficient to convert to numpy array first
    space_members = np.array(list(itertools.product(*ranges)), dtype=jnp.int32)
    return jnp.array(space_members)


def space_dimensions_from_bounds(
    bounds: tuple[jnp.ndarray, jnp.ndarray]
) -> tuple[int, ...]:
    """Return maximum size for each state dimension."""
    mins, maxs = bounds
    return tuple(jnp.array(maxs - mins + 1, dtype=jnp.int32))


def state_with_dimensions_to_index(
    state: jnp.ndarray, state_dimensions: tuple[int, ...]
) -> int:
    """Convert state vector to index if any state within the dimensions is valid."""
    return jnp.ravel_multi_index(tuple(state), state_dimensions, mode="wrap")
