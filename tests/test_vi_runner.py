"""Tests comparing ValueIterationRunner with original pymdptoolbox implementation."""

import mdptoolbox
import mdptoolbox.example
import numpy as np
import pytest

from mdpax.problems.forest import Forest
from mdpax.solvers.value_iteration import ValueIterationRunner


def test_vi_runner_matches_original():
    """Compare to the original pymdptoolbox implementation."""
    # Parameters
    S, r1, r2, p = 3, 4.0, 2.0, 0.1
    gamma = 0.95
    epsilon = 0.01

    # Get matrices from original implementation
    P_orig, R_orig = mdptoolbox.example.forest(S=S, r1=r1, r2=r2, p=p)

    # Run original value iteration
    vi_orig = mdptoolbox.mdp.ValueIteration(
        P_orig, R_orig, discount=gamma, epsilon=epsilon
    )
    vi_orig.run()

    # Run our value iteration
    forest = Forest(S=S, r1=r1, r2=r2, p=p)
    vi_runner = ValueIterationRunner(
        problem=forest,
        max_batch_size=32,  # Small problem, so small batch size is fine
        epsilon=epsilon,
        gamma=gamma,
    )
    results = vi_runner.run_value_iteration()
    print(results["output_info"])

    # Compare value functions
    # np.testing.assert_allclose(
    #    vi_orig.V,
    #    results['V'].values,
    #    rtol=1e-5,
    #    err_msg="Value functions don't match"
    # )

    # Compare policies
    np.testing.assert_array_equal(
        vi_orig.policy, vi_runner.policy, err_msg="Policies don't match"
    )


@pytest.mark.parametrize(
    "params",
    [
        {"S": 3, "r1": 4.0, "r2": 2.0, "p": 0.1, "gamma": 0.95, "epsilon": 0.001},
        {"S": 5, "r1": 10.0, "r2": 5.0, "p": 0.05, "gamma": 0.99, "epsilon": 0.001},
        {"S": 4, "r1": 2.0, "r2": 8.0, "p": 0.2, "gamma": 0.9, "epsilon": 0.001},
    ],
)
def test_vi_runner_different_parameters(params):
    """Test value iteration matches for different parameter settings."""
    # Split params
    mdp_params = {k: params[k] for k in ["S", "r1", "r2", "p"]}
    vi_params = {k: params[k] for k in ["gamma", "epsilon"]}

    # Get matrices from original implementation
    P_orig, R_orig = mdptoolbox.example.forest(**mdp_params)

    # Run original value iteration
    vi_orig = mdptoolbox.mdp.ValueIteration(
        P_orig, R_orig, discount=vi_params["gamma"], epsilon=vi_params["epsilon"]
    )
    vi_orig.run()

    # Run our value iteration
    forest = Forest(**mdp_params)
    vi_runner = ValueIterationRunner(problem=forest, max_batch_size=32, **vi_params)
    _ = vi_runner.run_value_iteration()

    # Compare value functions
    # np.testing.assert_allclose(
    #    vi_orig.V,
    #    results['V'].values,
    #    rtol=1e-5,
    #    err_msg="Value functions don't match"
    # )

    # Compare policies
    np.testing.assert_array_equal(
        vi_orig.policy, vi_runner.policy, err_msg="Policies don't match"
    )


def test_vi_runner_convergence():
    """Test that both implementations converge in similar number of iterations."""
    # Parameters
    S, r1, r2, p = 3, 4.0, 2.0, 0.1
    gamma = 0.95
    epsilon = 1e-8  # Tight convergence criterion

    # Original implementation
    P_orig, R_orig = mdptoolbox.example.forest(S=S, r1=r1, r2=r2, p=p)
    vi_orig = mdptoolbox.mdp.ValueIteration(
        P_orig, R_orig, discount=gamma, epsilon=epsilon
    )
    vi_orig.run()

    # Our implementation
    forest = Forest(S=S, r1=r1, r2=r2, p=p)
    vi_runner = ValueIterationRunner(
        problem=forest, max_batch_size=32, epsilon=epsilon, gamma=gamma
    )
    _ = vi_runner.run_value_iteration()

    # Check iterations are similar
    print(vi_orig.iter)
    print(vi_runner.iteration)
    assert abs(vi_orig.iter - vi_runner.iteration) <= 1


def test_vi_runner_performance():
    """Test that our implementation handles larger problems efficiently."""
    # Larger problem
    S = 1000
    forest = Forest(S=S, r1=4.0, r2=2.0, p=0.1)

    vi_runner = ValueIterationRunner(
        problem=forest, max_batch_size=128, epsilon=0.01, gamma=0.95
    )

    # Should complete in reasonable time
    results = vi_runner.run_value_iteration(max_iter=100)

    # Basic sanity checks
    assert results["V"].shape == (S, 1)
    assert results["policy"].shape == (S, 1)
    assert np.all(results["policy"].values >= 0)
    assert np.all(results["policy"].values <= 1)
