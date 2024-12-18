import mdptoolbox.example
import numpy as np
import pytest

import mdpax.mdptoolbox.example


class TestForest:
    """Test suite for the forest example MDP."""

    def _compare_outputs(self, mdp_P, mdp_R, jax_P, jax_R):
        """Helper method to compare outputs between mdptoolbox and jax versions."""
        jax_P = np.array(jax_P)
        jax_R = np.array(jax_R)
        np.testing.assert_allclose(mdp_P, jax_P)
        np.testing.assert_allclose(mdp_R, jax_R)

    def test_forest_default_params(self):
        """Test that forest() has same results as mdptoolbox with default parameters."""
        mdp_P, mdp_R = mdptoolbox.example.forest()
        jax_P, jax_R = mdpax.mdptoolbox.example.forest()
        self._compare_outputs(mdp_P, mdp_R, jax_P, jax_R)

    @pytest.mark.parametrize("S", [2, 5, 10])
    def test_forest_different_states(self, S):
        """Test that forest() works with different numbers of states."""
        mdp_P, mdp_R = mdptoolbox.example.forest(S=S)
        jax_P, jax_R = mdpax.mdptoolbox.example.forest(S=S)
        self._compare_outputs(mdp_P, mdp_R, jax_P, jax_R)

    @pytest.mark.parametrize("p", [0.01, 0.1, 0.5, 0.99])
    def test_forest_different_probabilities(self, p):
        """Test that forest() works with different fire probabilities."""
        mdp_P, mdp_R = mdptoolbox.example.forest(p=p)
        jax_P, jax_R = mdpax.mdptoolbox.example.forest(p=p)
        self._compare_outputs(mdp_P, mdp_R, jax_P, jax_R)

    @pytest.mark.parametrize("r1,r2", [(1, 1), (4, 2), (10, 5)])
    def test_forest_different_rewards(self, r1, r2):
        """Test that forest() works with different reward values."""
        mdp_P, mdp_R = mdptoolbox.example.forest(r1=r1, r2=r2)
        jax_P, jax_R = mdpax.mdptoolbox.example.forest(r1=r1, r2=r2)
        self._compare_outputs(mdp_P, mdp_R, jax_P, jax_R)

    def test_forest_sparse(self):
        """Test that sparse forest() returns correct values."""
        mdp_P, mdp_R = mdptoolbox.example.forest(is_sparse=True)
        jax_P, jax_R = mdpax.mdptoolbox.example.forest(is_sparse=True)

        # Convert JAX sparse to dense for comparison
        jax_P_dense = np.array(jax_P.todense())
        jax_R = np.array(jax_R)

        # Convert scipy sparse to dense
        mdp_P_dense = np.array([P.todense() for P in mdp_P])

        np.testing.assert_allclose(mdp_P_dense, jax_P_dense)
        np.testing.assert_allclose(mdp_R, jax_R)

    def test_invalid_params(self):
        """Test that invalid parameters raise appropriate errors."""
        invalid_params = [
            {"S": 1},  # S must be > 1
            {"r1": -1},  # rewards must be positive
            {"r2": -1},  # rewards must be positive
            {"p": -0.1},  # p must be in [0,1]
            {"p": 1.1},  # p must be in [0,1]
        ]

        for params in invalid_params:
            with pytest.raises(AssertionError):
                mdpax.mdptoolbox.example.forest(**params)
