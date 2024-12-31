import os

import numpy as np
import pandas as pd
import pytest

from mdpax.problems.mirjalili_perishable_platelet import MirjaliliPerishablePlatelet
from mdpax.solvers.periodic_value_iteration import PeriodicValueIteration


class TestPeriodicValueIterationPolicy:
    """Tests for Periodic Value Iteration solver using Mirjalili problem.

    Compares policies against reference results from implementation
    in viso_jax. These tests verify the solver produces correct policies
    for a moderately-sized problem with periodic demand (~5-20s runtime).

        Additional problem specific tests are in tests/problems/test_mirjalili.py
    """

    @pytest.mark.parametrize(
        "reported_policy_filename",
        [
            pytest.param(
                "mirjalili_m3_exp1_visojax.csv",
                id="mirjalili/m3/exp1",
            ),
        ],
    )
    def test_matches_reference_policy(
        self, tmpdir, shared_datadir, reported_policy_filename
    ):
        """Test policy matches results from original implementation.

        Verifies that our implementation produces the same policies as the
        viso_jax implementation for a platelet inventory problem
        with weekday-dependent demand patterns.
        """
        # Change working directory to avoid clutter
        os.chdir(tmpdir)

        problem = MirjaliliPerishablePlatelet()
        solver = PeriodicValueIteration(
            problem,
            gamma=0.95,
            max_iter=30,
            period=7,
            max_batch_size=5000,
            epsilon=1e-4,
        )
        result = solver.solve()
        policy = result.policy.reshape(-1)

        # Load in the reported policy
        reported_policy = pd.read_csv(
            f"{shared_datadir}/{reported_policy_filename}",
            index_col=0,
            header=0,
        )
        assert np.all(
            reported_policy.values.reshape(-1) == policy
        ), "Policy doesn't match"
