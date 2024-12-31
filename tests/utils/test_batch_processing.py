"""Tests for batch processing utilities."""

from unittest.mock import patch

import jax.numpy as jnp
import pytest

from mdpax.utils.batch_processing import BatchProcessor


@pytest.mark.parametrize(
    "n_states,state_dim,max_batch_size,expected_batch_size",
    [
        pytest.param(
            100,  # n_states less than default max_batch_size
            2,
            1024,
            100,  # batch_size should match n_states
            id="small_problem",
        ),
        pytest.param(
            2000,  # n_states greater than default max_batch_size
            3,
            1024,
            1024,  # batch_size should be capped at max
            id="large_problem",
        ),
        pytest.param(
            500,  # n_states less than custom max_batch_size
            2,
            512,
            500,  # batch_size should match n_states
            id="custom_max_small",
        ),
        pytest.param(
            1000,  # n_states greater than custom max_batch_size
            3,
            512,
            512,  # batch_size should be capped at custom max
            id="custom_max_large",
        ),
    ],
)
def test_batch_size_single_device(
    n_states, state_dim, max_batch_size, expected_batch_size
):
    """Test batch size calculation for single device.

    When running on a single device:
    - If n_states <= max_batch_size: batch_size = n_states
    - If n_states > max_batch_size: batch_size = max_batch_size
    """
    with patch("jax.devices") as mock_devices:
        mock_devices.return_value = [0]  # Single device
        processor = BatchProcessor(
            n_states=n_states,
            state_dim=state_dim,
            max_batch_size=max_batch_size,
        )
        assert processor.batch_size == expected_batch_size
        assert processor.n_devices == 1


@pytest.mark.parametrize(
    "n_states,state_dim,n_devices,max_batch_size,expected_batch_size",
    [
        pytest.param(
            1000,  # Small problem
            2,
            2,
            1024,
            500,  # n_states/n_devices
            id="even_split_small",
        ),
        pytest.param(
            4000,  # Large problem
            3,
            4,
            1024,
            1000,  # n_states/n_devices
            id="even_split_large",
        ),
        pytest.param(
            100,  # Very small problem
            2,
            4,
            1024,
            64,  # Minimum batch size (64)
            id="minimum_batch_size",
        ),
        pytest.param(
            10000,  # Very large problem
            3,
            2,
            512,
            512,  # Capped at max_batch_size
            id="maximum_batch_size",
        ),
    ],
)
def test_batch_size_multiple_devices(
    n_states, state_dim, n_devices, max_batch_size, expected_batch_size
):
    """Test batch size calculation for multiple devices.

    When running on multiple devices:
    - Minimum batch size is 64
    - Maximum batch size is user-provided max
    - Target is n_states/n_devices but within above bounds
    """
    with patch("jax.devices") as mock_devices:
        mock_devices.return_value = list(range(n_devices))
        processor = BatchProcessor(
            n_states=n_states,
            state_dim=state_dim,
            max_batch_size=max_batch_size,
        )
        assert processor.batch_size == expected_batch_size
        assert processor.n_devices == n_devices


def test_explicit_device_count():
    """Test that explicit pmap_device_count is respected."""
    with patch("jax.devices") as mock_devices:
        mock_devices.return_value = list(range(4))  # System has 4 devices
        processor = BatchProcessor(
            n_states=1000,
            state_dim=2,
            pmap_device_count=2,  # But we only want to use 2
        )
        assert processor.n_devices == 2


@pytest.mark.parametrize(
    "n_states,state_dim,n_devices,expected_shape",
    [
        pytest.param(
            100,  # Small problem
            2,
            1,
            (1, 1, 100, 2),  # Single batch on single device
            id="single_device_small",
        ),
        pytest.param(
            2000,  # Large problem
            3,
            1,
            (1, 2, 1024, 3),  # Two batches on single device
            id="single_device_large",
        ),
        pytest.param(
            500,  # Medium problem
            2,
            2,
            (2, 1, 250, 2),  # One batch per device
            id="multi_device_even",
        ),
        pytest.param(
            3000,  # Large problem
            3,
            4,
            (4, 1, 750, 3),  # One batch per device
            id="multi_device_large",
        ),
    ],
)
def test_prepare_batches(n_states, state_dim, n_devices, expected_shape):
    """Test prepare_batches shapes output correctly and handles padding.

    Verifies that:
    - Output has correct shape [n_devices, n_batches, batch_size, state_dim]
    - Padding is added when needed
    - Original values are preserved
    """
    with patch("jax.devices") as mock_devices:
        mock_devices.return_value = list(range(n_devices))
        processor = BatchProcessor(n_states=n_states, state_dim=state_dim)

        # Create test states
        states = jnp.ones((n_states, state_dim))

        # Prepare batches
        batched = processor.prepare_batches(states)

        # Check shape
        assert batched.shape == expected_shape

        # Check that original values are preserved (first n_states)
        unbatched = processor.unbatch_results(batched)
        assert jnp.array_equal(unbatched, states)


def test_batch_shape_property():
    """Test batch_shape property returns correct dimensions."""
    with patch("jax.devices") as mock_devices:
        mock_devices.return_value = list(range(2))
        processor = BatchProcessor(n_states=1000, state_dim=2)

        shape = processor.batch_shape
        assert len(shape) == 3
        assert shape[0] == processor.n_devices
        assert shape[1] == processor.n_batches
        assert shape[2] == processor.batch_size


def test_unbatch_results_with_extra_dims():
    """Test unbatch_results handles additional dimensions correctly."""
    with patch("jax.devices") as mock_devices:
        mock_devices.return_value = [0]  # Single device
        processor = BatchProcessor(n_states=100, state_dim=2)

        # Create batched results with an extra dimension
        # Shape: [n_devices, n_batches, batch_size, extra_dim]
        batched = jnp.ones((1, 1, 100, 5))

        # Unbatch
        unbatched = processor.unbatch_results(batched)

        # Check shape: [n_states, extra_dim]
        assert unbatched.shape == (100, 5)
