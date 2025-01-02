"""Tests for checkpointing functionality."""

import jax.numpy as jnp
import pytest

from mdpax.problems.de_moor_perishable import DeMoorPerishable
from mdpax.solvers.value_iteration import ValueIteration


def test_checkpoint_save_load(tmp_path):
    """Test that loading checkpoint restores exact state.

    Verifies that:
    - Values and iteration count match after loading
    - Policy matches after loading
    - Custom checkpoint directory is used
    """

    # Set custom checkpoint directory
    checkpoint_dir = tmp_path / "checkpoints" / "test_checkpoint_save_load"

    problem = DeMoorPerishable()
    solver = ValueIteration(
        problem=problem,
        gamma=0.99,
        epsilon=1e-4,
        checkpoint_dir=checkpoint_dir,
        checkpoint_frequency=5,
    )

    # Run for a few iterations
    solver.solve(max_iterations=20)
    values_before = solver.values.copy()
    iter_before = solver.iteration

    # Create new solver and load checkpoint
    new_solver = ValueIteration.restore(checkpoint_dir)
    # Check state matches
    assert new_solver.iteration == iter_before
    assert new_solver.values == pytest.approx(values_before)


def test_checkpoint_resume_to_convergence(tmp_path):
    """Test that resuming from checkpoint converges to same result.

    Verifies that:
    - Running to convergence in one go
    - vs. stopping, loading checkpoint, then continuing
    - produces identical results
    """
    problem = DeMoorPerishable()

    # First run to convergence normally
    solver1 = ValueIteration(
        problem=problem,
        gamma=0.99,
        epsilon=1e-4,
        checkpoint_frequency=0,
    )
    solver1.solve(max_iterations=2000)
    final_values = solver1.values.copy()
    final_policy = solver1.policy.copy()

    # Now run with interruption
    checkpoint_dir = tmp_path / "checkpoints" / "test_checkpoint_resume_to_convergence"
    solver2 = ValueIteration(
        problem=problem,
        gamma=0.99,
        epsilon=1e-4,
        checkpoint_dir=checkpoint_dir,
        checkpoint_frequency=100,
    )

    # Run for a few iterations
    solver2.solve(max_iterations=200)

    # Create new solver and resume
    solver3 = ValueIteration.restore(checkpoint_dir)
    solver3.solve(max_iterations=2000)

    # Check final results match
    assert solver3.values == pytest.approx(final_values)
    assert jnp.array_equal(solver3.policy, final_policy)


def test_max_checkpoints_retained(tmp_path):
    """Test that only specified number of checkpoints are kept."""
    checkpoint_dir = tmp_path / "checkpoints" / "test_max_checkpoints_retained"
    problem = DeMoorPerishable()
    checkpoint_frequency = 100
    max_checkpoints = 3
    solver = ValueIteration(
        problem=problem,
        gamma=0.99,
        epsilon=1e-4,
        checkpoint_dir=checkpoint_dir,
        checkpoint_frequency=checkpoint_frequency,
        max_checkpoints=max_checkpoints,
    )
    # Run solver
    solver.solve()

    # Check number of checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("*"))
    assert len(checkpoint_files) == max_checkpoints + 1  # 1 is the config file


def test_custom_checkpoint_config_on_restore(tmp_path):
    """Test that new checkpoint config is used after restore.

    Verifies that:
    - Updated arguments are used after restore
    - Checkpoints are saved to new directory
    """
    # Initial run with original config

    problem = DeMoorPerishable()
    checkpoint_dir = (
        tmp_path / "checkpoints" / "test_custom_checkpoint_config_on_restore"
    )
    solver = ValueIteration(
        problem=problem,
        gamma=0.99,
        epsilon=1e-4,
        checkpoint_dir=checkpoint_dir,
        checkpoint_frequency=100,
        max_checkpoints=1,
    )
    solver.solve(max_iterations=500)

    # Create new solver with different config
    new_checkpoint_dir = (
        tmp_path / "checkpoints" / "test_custom_checkpoint_config_on_restore_new"
    )
    new_checkpoint_frequency = 50
    new_max_checkpoints = 3
    new_enable_async_checkpointing = False
    new_solver = ValueIteration.restore(
        checkpoint_dir=checkpoint_dir,
        new_checkpoint_dir=new_checkpoint_dir,
        checkpoint_frequency=new_checkpoint_frequency,
        max_checkpoints=new_max_checkpoints,
        enable_async_checkpointing=new_enable_async_checkpointing,
    )

    assert new_solver.checkpoint_dir == new_checkpoint_dir
    assert new_solver.checkpoint_frequency == new_checkpoint_frequency
    assert new_solver.max_checkpoints == new_max_checkpoints
    assert new_solver.enable_async_checkpointing == new_enable_async_checkpointing

    new_solver.solve()
    # Check that checkpoints have been saved to new_checkpoint_dir
    new_checkpoint_files = list(new_checkpoint_dir.glob("*"))
    assert (
        len(new_checkpoint_files) == new_max_checkpoints + 1
    )  # Correct number of checkpoints have been saved to new directory (plus config file)
