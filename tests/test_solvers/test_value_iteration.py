import os

import jax
import jax.numpy as jnp
import mdptoolbox.example
import numpy as np
import pandas as pd
import pytest

from mdpax.problems.forest import Forest
from mdpax.problems.perishable_inventory.de_moor_single_product import (
    DeMoorSingleProductPerishable,
)
from mdpax.solvers.value_iteration import ValueIteration

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            {"S": 3, "r1": 4.0, "r2": 2.0, "p": 0.1},
            id="forest/default",
        ),
        pytest.param(
            {"S": 5, "r1": 10.0, "r2": 5.0, "p": 0.05},
            id="forest/large_rewards",
        ),
        pytest.param(
            {"S": 4, "r1": 2.0, "r2": 8.0, "p": 0.2},
            id="forest/high_risk",
        ),
    ],
)
def test_forest_matches_pymdptoolbox(params):
    """Test policy matches pymdptoolbox implementation.

    Verifies that our implementation produces the same policies and values as
    pymdptoolbox, with the same number of iterations, for the Forest Management
    problem under different parameter settings.
    """
    # Get matrices from both implementations
    P, R = mdptoolbox.example.forest(**params)
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9, epsilon=0.01)
    vi.run()
    pymdptoolbox_policy = np.array(vi.policy)
    pymdptoolbox_values = np.array(vi.V)
    pymdptoolbox_iter = vi.iter

    forest = Forest(**params)
    solver = ValueIteration(forest, gamma=0.9, epsilon=0.01, verbose=0)
    result = solver.solve()
    mdpax_policy = result.policy.reshape(-1)
    mdpax_values = result.values.reshape(-1)
    mdpax_iter = result.info.iteration

    # Policy comparison (integer arrays)
    assert jnp.array_equal(
        mdpax_policy, pymdptoolbox_policy
    ), "Policy doesn't match pymdptoolbox"

    # Values comparison (float arrays)
    assert mdpax_values == pytest.approx(
        pymdptoolbox_values, rel=1e-5
    ), "Values don't match pymdptoolbox"

    # Iteration comparison (integers)
    assert (
        mdpax_iter == pymdptoolbox_iter
    ), "Number of iterations doesn't match pymdptoolbox"


@pytest.mark.parametrize(
    "issuing_policy,reported_policy_filename",
    [
        pytest.param(
            "lifo",
            "de_moor_perishable_m2_exp1_reported_policy.csv",
            id="de_moor/m2/exp1",
        ),
        pytest.param(
            "fifo",
            "de_moor_perishable_m2_exp2_reported_policy.csv",
            id="de_moor/m2/exp2",
        ),
    ],
)
def test_de_moor_matches_paper(
    tmpdir, shared_datadir, issuing_policy, reported_policy_filename
):
    """Test policy matches results from original paper.

    Verifies that our implementation produces the same policies as reported
    in Figure 3 of De Moor et al. (2022) for both LIFO and FIFO cases.
    """
    # Change working directory to avoid clutter
    os.chdir(tmpdir)

    problem = DeMoorSingleProductPerishable(issue_policy=issuing_policy)
    solver = ValueIteration(problem, gamma=0.99, epsilon=1e-5, verbose=0)
    result = solver.solve(max_iterations=5000)
    policy = result.policy.reshape(-1)

    vi_policy = pd.DataFrame(policy)
    # Post-process policy to match reported form
    # Including clipping so that only includes stock-holding up to 8 units per age
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

    # Policy comparison (integer arrays)
    assert jnp.array_equal(
        vi_policy.values, reported_policy.values
    ), "Policy doesn't match"


@pytest.mark.parametrize(
    "max_useful_life,lead_time,issue_policy,reported_policy_filename",
    [
        pytest.param(
            3,
            2,
            "lifo",
            "de_moor_perishable_m3_exp5_visojax.csv",
            id="de_moor/m3/exp5",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_de_moor_matches_viso_jax_reference_policy(
    shared_datadir,
    max_useful_life,
    lead_time,
    issue_policy,
    reported_policy_filename,
):
    """Test policy matches reference policy from viso_jax."""
    problem = DeMoorSingleProductPerishable(
        max_useful_life=max_useful_life,
        lead_time=lead_time,
        issue_policy=issue_policy,
    )
    solver = ValueIteration(problem, gamma=0.99, epsilon=1e-5, verbose=0)
    result = solver.solve(max_iterations=5000)
    policy = result.policy.reshape(-1)
    reported_policy = pd.read_csv(
        f"{shared_datadir}/{reported_policy_filename}",
        index_col=0,
        header=0,
    )
    # Policy comparison (integer arrays)
    assert jnp.array_equal(
        policy, reported_policy.values.flatten()
    ), "Policy doesn't match"
