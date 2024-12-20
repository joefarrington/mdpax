"""Tests comparing ValueIterationRunner with original pymdptoolbox implementation."""

import mdptoolbox
import mdptoolbox.example
import numpy as np
import pytest

from mdpax.problems.forest import Forest
from mdpax.solvers.value_iteration import ValueIteration


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
    vi_runner = ValueIteration(
        problem=forest,
        batch_size=32,  # Small problem, so small batch size is fine
        epsilon=epsilon,
        gamma=gamma,
    )
    _, policy = vi_runner.solve()

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
    vi_runner = ValueIteration(problem=forest, batch_size=32, **vi_params)
    _, policy = vi_runner.solve()

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
    vi_runner = ValueIteration(
        problem=forest, batch_size=32, epsilon=epsilon, gamma=gamma
    )
    _, _ = vi_runner.solve()

    # Check iterations are similar
    print(vi_orig.iter)
    print(vi_runner.iteration)
    assert abs(vi_orig.iter - vi_runner.iteration) <= 1


def test_vi_runner_performance():
    """Test that our implementation handles larger problems efficiently."""
    # Larger problem
    S = 1000
    forest = Forest(S=S, r1=4.0, r2=2.0, p=0.1)

    vi_runner = ValueIteration(
        problem=forest, batch_size=128, epsilon=0.01, gamma=0.95, max_iter=100
    )

    # Should complete in reasonable time
    _, _ = vi_runner.solve()

    # Basic sanity checks
    assert vi_runner.values.shape == (S, 1)
    assert vi_runner.policy.shape == (S, 1)
    assert np.all(vi_runner.policy.values >= 0)
    assert np.all(vi_runner.policy.values <= 1)
