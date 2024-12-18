import os

import jax
import numpy as np
import pandas as pd
import pytest

from mdpax.problems.hendrix_perishable_substitution_two_product import (
    HendrixPerishableSubstitutionTwoProduct,
)
from mdpax.solvers.value_iteration import ValueIterationRunner


# Compare policy output from new implementation with vio jax
class TestPolicy:
    @pytest.mark.parametrize(
        "issuing_policy,reported_policy_filename",
        [
            pytest.param(
                "m2_exp1",
                "hendrix_m2_exp1_visojax.csv",
                id="m2/exp1",
            ),
        ],
    )
    def test_policy_same_as_reported(
        self, tmpdir, shared_datadir, issuing_policy, reported_policy_filename
    ):
        jax.config.update("jax_enable_x64", True)
        # Change working directory to avoid clutter
        os.chdir(tmpdir)

        problem = HendrixPerishableSubstitutionTwoProduct()
        vi_runner = ValueIterationRunner(
            problem, max_batch_size=5000, gamma=1.0, epsilon=1e-5
        )
        vi_output = vi_runner.run_value_iteration(max_iter=100)
        # Load in the reported policy
        reported_policy = pd.read_csv(
            f"{shared_datadir}/{reported_policy_filename}",
            index_col=0,
            header=0,
        )
        print(np.sum(reported_policy.values != vi_output["policy"].values))
        assert np.all(reported_policy.values == vi_output["policy"].values)
