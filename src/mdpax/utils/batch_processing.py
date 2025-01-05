"""Utilities for batch processing of state spaces."""

from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from loguru import logger


class BatchProcessor:
    """Handles batching and padding of state spaces for parallel processing.

    This class manages the batching of states for efficient parallel processing across
    multiple devices. It handles:
    - Determining batch sizes based on problem size and available devices
    - Padding state arrays to fit batch dimensions
    - Reshaping arrays for device distribution
    - Removing padding and batching from results

    Args:
        n_states: Total number of states in the problem.
        state_dim: Dimensionality of each state vector.
        max_batch_size: Maximum allowed size for batches. Defaults to 1024.
        pmap_device_count: Number of JAX devices to use. If None, uses all available.

    Attributes:
        n_states (int): Total number of states in the problem.
        state_dim (int): Dimensionality of each state vector.
        n_devices (int): Number of devices being used.
        batch_size (int): Actual batch size after adjusting for problem size and devices.
        n_pad (int): Number of padding elements added.
        n_batches (int): Number of batches per device.

    Example:
        >>> n_states = 1000
        >>> state_dim = 3
        >>> processor = BatchProcessor(n_states, state_dim)
        >>> states = jnp.zeros((n_states, state_dim))
        >>> batched = processor.prepare_batches(states)  # Shape: [n_devices, n_batches, batch_size, 3]
        >>> results = some_operation(batched)
        >>> unbatched = processor.unbatch_results(results)  # Shape: [1000, ...]
    """

    def __init__(
        self,
        n_states: int,
        state_dim: int,
        max_batch_size: int = 1024,
        pmap_device_count: Int[Array, ""] = None,
    ) -> None:

        self.n_states = n_states
        self.state_dim = state_dim

        # Setup device information
        self.n_devices = (
            len(jax.devices()) if pmap_device_count is None else pmap_device_count
        )

        # Calculate states per device (ceiling division to ensure all states covered)
        states_per_device = (n_states + self.n_devices - 1) // self.n_devices

        # Calculate appropriate batch size
        if self.n_devices == 1:
            # Single device - clip to problem size
            self.batch_size = min(max_batch_size, n_states)
        else:
            # Multiple devices - ensure even distribution
            self.batch_size = min(
                max_batch_size,  # user provided/default max
                max(64, states_per_device),  # ensure minimum batch size
            )

        # Calculate batches per device
        if states_per_device <= self.batch_size:
            # Small problem - single batch per device
            self.n_batches = 1
        else:
            # Multiple batches needed per device
            self.n_batches = (
                states_per_device + self.batch_size - 1
            ) // self.batch_size

        # Calculate padding needed
        total_size = self.n_devices * self.n_batches * self.batch_size
        self.n_pad = total_size - n_states

        logger.debug(f"Batch processor initialized with {self.n_devices} device(s)")
        logger.debug(f"Batch size: {self.batch_size}")
        logger.debug(f"Number of batches per device: {self.n_batches}")
        logger.debug(f"Padding elements: {self.n_pad}")

    def prepare_batches(
        self, states: Float[Array, "n_states state_dim"]
    ) -> Float[Array, "n_devices n_batches batch_size state_dim"]:
        """Prepare states for batch processing.

        Pads the state array if needed and reshapes it for distribution across devices.

        Args:
            states: Array of states with shape [n_states, state_dim].

        Returns:
            Array of batched and padded states with shape
            [n_devices, n_batches, batch_size, state_dim].
        """
        # Pad if needed
        if self.n_pad > 0:
            states = jnp.vstack(
                [states, jnp.zeros((self.n_pad, self.state_dim), dtype=states.dtype)]
            )

        return states.reshape(
            self.n_devices, self.n_batches, self.batch_size, self.state_dim
        )

    def unbatch_results(
        self, batched_results: Float[Array, "n_devices n_batches batch_size *dims"]
    ) -> Float[Array, "n_states *dims"]:
        """Remove batching and padding from results.

        Args:
            batched_results: Results from batch processing with shape
                [n_devices, n_batches, batch_size, \\*dims] where \\*dims are
                any additional dimensions from the operation.

        Returns:
            Array of unbatched and unpadded results with shape [n_states, \\*dims].
        """
        # Reshape to flatten batch dimensions
        results = jnp.reshape(batched_results, (-1, *batched_results.shape[3:]))

        # Remove padding if needed
        if self.n_pad > 0:
            return results[: -self.n_pad]
        return results

    @property
    def batch_shape(self) -> Tuple[int, int, int]:
        """Get the shape of batched data.

        Returns:
            Tuple of (n_devices, n_batches, batch_size) indicating how the data
            will be distributed across devices and batches.
        """
        return (self.n_devices, self.n_batches, self.batch_size)
