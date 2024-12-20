import os

import jax
import numpy as np
import pandas as pd
import pytest

from mdpax.problems.de_moor_perishable import DeMoorPerishable
from mdpax.solvers.value_iteration import ValueIteration


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
        vi_runner = ValueIteration(problem, gamma=0.99, max_iter=5000, epsilon=1e-5)
        _, policy = vi_runner.solve()

        vi_policy = pd.DataFrame(policy)
        # Post-process policy to match reported form
        # Including clipping so that only includes stock-holding up to 8 units per agre
        vi_policy.columns = ["order_quantity"]
        vi_policy["Units in stock age 2"] = [
            int(x[1]) for x in np.array(problem.state_space)
        ]
        vi_policy["Units in stock age 1"] = [
            int(x[0]) for x in np.array(problem.state_space)
        ]
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
