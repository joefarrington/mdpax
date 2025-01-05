import os

import jax
import jax.numpy as jnp
import pandas as pd
import pytest

from mdpax.problems.perishable_inventory.mirjalili_platelet import (
    MirjaliliPlateletPerishable,
)
from mdpax.solvers.periodic_value_iteration import PeriodicValueIteration

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "reported_policy_filename",
    [
        pytest.param(
            "mirjalili_m3_exp1_visojax.csv",
            id="mirjalili/m3/exp1",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_matches_reference_policy(tmpdir, shared_datadir, reported_policy_filename):
    """Test policy matches results from original implementation.

    Verifies that our implementation produces the same policies as the
    viso_jax implementation for a platelet inventory problem
    with weekday-dependent demand patterns.
    """
    # Change working directory to avoid clutter
    os.chdir(tmpdir)

    problem = MirjaliliPlateletPerishable()
    solver = PeriodicValueIteration(
        problem,
        gamma=0.95,
        period=7,
        max_batch_size=5000,
        epsilon=1e-4,
    )
    result = solver.solve(max_iterations=30)
    policy = result.policy.reshape(-1)

    # Load in the reported policy
    reported_policy_df = pd.read_csv(
        f"{shared_datadir}/{reported_policy_filename}",
        index_col=0,
        header=0,
    )
    reported_policy = jnp.array(reported_policy_df.values.reshape(-1))
    assert jnp.array_equal(reported_policy, policy)
