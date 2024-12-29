"""Utilities for batch processing of state spaces."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from loguru import logger


class BatchProcessor:
    """Handles batching and padding of state spaces for parallel processing.

    This class manages the batching of states for efficient parallel processing across
    multiple devices, handling padding and reshaping as needed.

    Args:
        n_states: Total number of states
        state_dim: Dimensionality of each state
        max_batch_size: Maximum size for batches
        jit_device_count: Number of available JAX devices (defaults to all)

    Attributes:
        n_states: Total number of states
        state_dim: Dimensionality of each state
        n_devices: Number of devices being used
        batch_size: Actual batch size after adjusting for problem size and devices
        n_pad: Number of padding elements added
        n_batches: Number of batches per device
    """

    def __init__(
        self,
        n_states: int,
        state_dim: int,
        max_batch_size: int = 1024,
        jit_device_count: int = None,
    ):
        """Initialize the batch processor."""
        self.n_states = n_states
        self.state_dim = state_dim

        # Setup device information
        self.n_devices = (
            len(jax.devices()) if jit_device_count is None else jit_device_count
        )

        # Calculate appropriate batch size
        if self.n_devices == 1:
            # Single device - clip to problem size
            self.batch_size = min(max_batch_size, n_states)
        else:
            # Multiple devices - ensure even distribution
            states_per_device = n_states // self.n_devices
            self.batch_size = min(
                max_batch_size,  # user provided/default max
                max(64, states_per_device),  # ensure minimum batch size
            )

        # Calculate batching parameters
        if n_states <= self.batch_size:
            # Small problem - single batch
            self.n_batches = 1
        else:
            # Multiple batches needed
            self.n_batches = (n_states + self.batch_size - 1) // self.batch_size

        # Calculate padding needed
        total_size = self.n_devices * self.n_batches * self.batch_size
        self.n_pad = total_size - n_states

        logger.debug(f"Batch processor initialized with {self.n_devices} devices")
        logger.debug(f"Batch size: {self.batch_size}")
        logger.debug(f"Number of batches per device: {self.n_batches}")
        logger.debug(f"Padding elements: {self.n_pad}")

    def prepare_batches(self, states: chex.Array) -> chex.Array:
        """Prepare states for batch processing.

        Args:
            states: Array of states [n_states, state_dim]

        Returns:
            Batched and padded states [n_devices, n_batches, batch_size, state_dim]
        """
        # Pad if needed
        if self.n_pad > 0:
            states = jnp.vstack(
                [states, jnp.zeros((self.n_pad, self.state_dim), dtype=states.dtype)]
            )

        # Reshape to standard format: (devices, batches, batch_size, state_dim)
        return states.reshape(
            self.n_devices, self.n_batches, self.batch_size, self.state_dim
        )

    def unbatch_results(self, batched_results: chex.Array) -> chex.Array:
        """Remove batching and padding from results.

        Args:
            batched_results: Results from batch processing
                Shape depends on operation but first dimensions match batch shape

        Returns:
            Unbatched and unpadded results [n_states, ...]
        """
        # Reshape to flatten batch dimensions
        results = jnp.reshape(batched_results, (-1, *batched_results.shape[3:]))

        # Remove padding if needed
        if self.n_pad > 0:
            return results[: -self.n_pad]
        return results

    @property
    def batch_shape(self) -> Tuple[int, int, int]:
        """Get the shape of batched data (n_devices, n_batches, batch_size)."""
        return (self.n_devices, self.n_batches, self.batch_size)
