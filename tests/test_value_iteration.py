"""Tests comparing JAX Value Iteration with original pymdptoolbox implementation."""

import mdptoolbox.example
import mdptoolbox.mdp
import numpy as np
import pytest

from mdpax.mdptoolbox.mdp import ValueIteration


def compare_vi_results(P, R, discount, epsilon=0.01, max_iter=1000):
    """Compare results between original and JAX implementations."""
    # Run original implementation
    vi_orig = mdptoolbox.mdp.ValueIteration(
        P, R, discount, epsilon=epsilon, max_iter=max_iter
    )
    vi_orig.run()

    # Run JAX implementation
    vi_jax = ValueIteration(P, R, discount, epsilon=epsilon, max_iter=max_iter)
    vi_jax.run()

    # Convert JAX arrays to numpy for comparison
    jax_V = np.array(vi_jax.V)
    jax_policy = np.array(vi_jax.policy)

    # Compare results
    np.testing.assert_allclose(
        vi_orig.V, jax_V, rtol=1e-5, err_msg="Value functions don't match"
    )
    np.testing.assert_array_equal(
        vi_orig.policy, jax_policy, err_msg="Policies don't match"
    )

    return vi_orig, vi_jax


@pytest.mark.parametrize("S", [3, 5, 10])
def test_forest_different_sizes(S):
    """Test both implementations give same results for different forest sizes."""
    P, R = mdptoolbox.example.forest(S=S)
    compare_vi_results(P, R, discount=0.9)


@pytest.mark.parametrize("discount", [0.1, 0.5, 0.9, 0.99])
def test_forest_different_discounts(discount):
    """Test both implementations give same results for different discount rates."""
    P, R = mdptoolbox.example.forest()
    compare_vi_results(P, R, discount=discount)


@pytest.mark.parametrize("p", [0.01, 0.1, 0.5])
def test_forest_different_fire_probabilities(p):
    """Test both implementations give same results for different fire probabilities."""
    P, R = mdptoolbox.example.forest(p=p)
    compare_vi_results(P, R, discount=0.9)


@pytest.mark.parametrize("r1,r2", [(2, 2), (4, 2), (10, 2)])
def test_forest_different_rewards(r1, r2):
    """Test both implementations give same results for different reward structures."""
    P, R = mdptoolbox.example.forest(r1=r1, r2=r2)
    compare_vi_results(P, R, discount=0.9)


def test_forest_convergence_speed():
    """Compare number of iterations needed for convergence."""
    P, R = mdptoolbox.example.forest()
    vi_orig, vi_jax = compare_vi_results(P, R, discount=0.9, epsilon=1e-8)

    print("\nIterations needed:")
    print(f"Original: {vi_orig.iter}")
    print(f"JAX: {vi_jax.iter}")

    assert (
        abs(vi_orig.iter - vi_jax.iter) <= 1
    ), "Implementations took significantly different numbers of iterations"


def test_forest_timing():
    """Compare execution time (excluding compilation)."""
    P, R = mdptoolbox.example.forest(S=50)  # Larger problem for timing

    # First run to compile JAX code
    vi_jax = ValueIteration(P, R, 0.9)
    vi_jax.run()

    # Now time both implementations
    vi_orig = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    vi_jax = ValueIteration(P, R, 0.9)

    vi_orig.run()
    vi_jax.run()

    print("\nExecution time (seconds):")
    print(f"Original: {vi_orig.time:.6f}")
    print(f"JAX: {vi_jax.time:.6f}")
