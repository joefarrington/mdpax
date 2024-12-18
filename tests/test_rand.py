import mdptoolbox.example
import numpy as np
import pytest

from mdpax.problems.rand import RandDense


def test_rand_matches_original():
    """Test that our RandDense implementation matches the original."""
    # Parameters to test
    S, A = 4, 3
    np.random.seed(42)  # Set seed for our version
    problem = RandDense(states=S, actions=A)
    P_new = np.array(problem.P)
    R_new = np.array(problem.R)

    # Get matrices from original implementation
    np.random.seed(42)  # Reset seed for original version
    P_orig, R_orig = mdptoolbox.example.rand(S=S, A=A)

    # Compare transition matrices
    np.testing.assert_allclose(
        P_orig, P_new, rtol=1e-5, err_msg="Transition matrices don't match"
    )

    # Compare reward matrices
    np.testing.assert_allclose(
        R_orig, R_new, rtol=1e-5, err_msg="Reward matrices don't match"
    )


@pytest.mark.parametrize(
    "params",
    [
        {"S": 3, "A": 2},  # Small
        {"S": 5, "A": 4},  # Medium
        {"S": 10, "A": 3},  # Larger states
        {"S": 4, "A": 8},  # Larger actions
    ],
)
def test_rand_different_sizes(params):
    """Test that matrices match for different problem sizes."""
    np.random.seed(42)
    problem = RandDense(states=params["S"], actions=params["A"])
    P_new = np.array(problem.P)
    R_new = np.array(problem.R)

    np.random.seed(42)
    P_orig, R_orig = mdptoolbox.example.rand(**params)

    np.testing.assert_allclose(P_orig, P_new, rtol=1e-5)
    np.testing.assert_allclose(R_orig, R_new, rtol=1e-5)


def test_rand_with_mask():
    """Test that matrices match when using a mask."""
    S, A = 4, 3
    # Create a simple mask
    mask = np.random.random((S, S)) > 0.5

    np.random.seed(42)
    problem = RandDense(states=S, actions=A, mask=mask)
    P_new = np.array(problem.P)
    R_new = np.array(problem.R)

    np.random.seed(42)
    P_orig, R_orig = mdptoolbox.example.rand(S=S, A=A, mask=mask)

    np.testing.assert_allclose(P_orig, P_new, rtol=1e-5)
    np.testing.assert_allclose(R_orig, R_new, rtol=1e-5)


def test_properties():
    """Test basic properties of the implementation."""
    problem = RandDense(states=4, actions=3)

    # Check dimensions
    assert problem.P.shape == (3, 4, 4)  # [A, S, S]
    assert problem.R.shape == (3, 4, 4)  # [A, S, S]

    # Check probabilities sum to 1
    for a in range(3):
        for s in range(4):
            assert np.allclose(problem.P[a, s].sum(), 1.0)

    # Check rewards are in [-1, 1]
    assert np.all(problem.R >= -1)
    assert np.all(problem.R <= 1)
