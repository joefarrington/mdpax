import os

import jax.numpy as jnp
import pandas as pd
import pytest

from mdpax.problems.perishable_inventory.hendrix_two_product import (
    HendrixTwoProductPerishable,
)
from mdpax.solvers.relative_value_iteration import RelativeValueIteration


class TestRelativeValueIterationPolicy:
    """Tests for Relative Value Iteration solver using Hendrix problem.

    Compares policies against reference results from implementation
    in viso_jax. These tests verify the solver produces correct policies
    for a moderately-sized problem with substitution (~5-20s runtime).

    Additional problem specific tests are in tests/problems/test_hendrix.py
    """

    @pytest.mark.parametrize(
        "reported_policy_filename",
        [
            pytest.param(
                "hendrix_m2_exp1_visojax.csv",
                id="hendrix/m2/exp1",
            ),
        ],
    )
    def test_matches_reference_policy(
        self, tmpdir, shared_datadir, reported_policy_filename
    ):
        """Test policy matches results from original implementation.

        Verifies that our implementation produces the same policies as the
        viso_jax implementation for a two-product substitution problem.
        """
        # Change working directory to avoid clutter
        os.chdir(tmpdir)

        problem = HendrixTwoProductPerishable()
        solver = RelativeValueIteration(
            problem,
            epsilon=1e-4,
        )
        result = solver.solve()
        policy = result.policy.reshape(-1)

        # Load in the reported policy
        reported_policy_df = pd.read_csv(
            f"{shared_datadir}/{reported_policy_filename}",
            index_col=0,
            header=0,
        )
        reported_policy = jnp.array(reported_policy_df.values.reshape(-1))
        assert jnp.array_equal(reported_policy, policy)
