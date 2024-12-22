import os

import jax
import numpy as np
import pandas as pd
import pytest

from mdpax.problems.hendrix_perishable_substitution_two_product import (
    HendrixPerishableSubstitutionTwoProduct,
)
from mdpax.solvers.relative_value_iteration import RelativeValueIteration


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
        vi_runner = RelativeValueIteration(
            problem, batch_size=5000, epsilon=1e-5, max_iter=100
        )
        _, policy, _ = vi_runner.solve()
        policy = np.array(policy)
        # Load in the reported policy
        reported_policy = pd.read_csv(
            f"{shared_datadir}/{reported_policy_filename}",
            index_col=0,
            header=0,
        )
        print(np.sum(reported_policy.values != policy))
        assert np.all(reported_policy.values == policy)
