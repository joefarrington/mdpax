import os

import jax
import numpy as np
import pandas as pd
import pytest

from mdpax.problems.de_moor_perishable import DeMoorPerishable
from mdpax.solvers.value_iteration import ValueIterationRunner


# Compare policy output by DeMoorPerishableVIR with policies
# printed in Fig 3 of De Moor et al (2022)
class TestPolicy:
    @pytest.mark.parametrize(
        "issuing_policy,reported_policy_filename",
        [
            pytest.param(
                "lifo",
                "de_moor_perishable_m2_exp1_reported_policy.csv",
                id="m2/exp1",
            ),
            pytest.param(
                "fifo",
                "de_moor_perishable_m2_exp2_reported_policy.csv",
                id="m2/exp2",
            ),
        ],
    )
    def test_policy_same_as_reported(
        self, tmpdir, shared_datadir, issuing_policy, reported_policy_filename
    ):
        jax.config.update("jax_enable_x64", True)
        # Change working directory to avoid clutter
        os.chdir(tmpdir)

        problem = DeMoorPerishable(issue_policy=issuing_policy)
        vi_runner = ValueIterationRunner(
            problem, max_batch_size=150, gamma=0.99, epsilon=1e-5
        )
        vi_output = vi_runner.run_value_iteration(max_iter=10000)

        # Post-process policy to match reported form
        # Including clipping so that only includes stock-holding up to 8 units per agre
        vi_policy = vi_output["policy"].reset_index()
        vi_policy.columns = ["state", "order_quantity"]
        vi_policy["Units in stock age 2"] = [(x)[1] for x in vi_policy["state"]]
        vi_policy["Units in stock age 1"] = [(x)[0] for x in vi_policy["state"]]
        vi_policy = vi_policy.pivot(
            index="Units in stock age 1",
            columns="Units in stock age 2",
            values="order_quantity",
        )
        vi_policy = vi_policy.loc[list(range(9)), list(range(9))].sort_index(
            ascending=False
        )

        # Load in the reported policy
        reported_policy = pd.read_csv(
            f"{shared_datadir}/{reported_policy_filename}",
            index_col=0,
            header=0,
        )

        assert np.all(vi_policy.values == reported_policy.values)
