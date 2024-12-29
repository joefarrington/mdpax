import os

import numpy as np
import pandas as pd
import pytest

from mdpax.problems.mirjalili_perishable_platelet import MirjaliliPerishablePlatelet
from mdpax.solvers.periodic_value_iteration import PeriodicValueIteration


# Compare policy output from new implementation with joefarrington/viso_jax
class TestPolicy:
    @pytest.mark.parametrize(
        "reported_policy_filename",
        [
            pytest.param(
                "mirjalili_m3_exp1_visojax.csv",
                id="m3/exp1",
            ),
        ],
    )
    def test_policy_same_as_reported(
        self, tmpdir, shared_datadir, reported_policy_filename
    ):
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
        assert np.all(reported_policy.values.reshape(-1) == policy)
