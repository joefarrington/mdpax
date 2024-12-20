import os

import jax
import numpy as np
import pandas as pd
import pytest

from mdpax.problems.mirjalili_perishable_platelet import MirjaliliPerishablePlatelet
from mdpax.solvers.value_iteration import ValueIteration


# Compare policy output from new implementation with vio jax
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
        jax.config.update("jax_enable_x64", True)
        # Change working directory to avoid clutter
        os.chdir(tmpdir)

        problem = MirjaliliPerishablePlatelet()
        vi_runner = ValueIteration(
            problem, batch_size=200, gamma=0.95, epsilon=1e-5, max_iter=20
        )
        _, policy = vi_runner.solve()
        # Load in the reported policy
        reported_policy = pd.read_csv(
            f"{shared_datadir}/{reported_policy_filename}",
            index_col=0,
            header=0,
        )
        assert np.all(reported_policy.values.reshape(-1) == policy)
