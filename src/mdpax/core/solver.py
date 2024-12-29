"""Base class for MDP solvers."""

import sys
from abc import ABC, abstractmethod

import chex
import jax
import jax.numpy as jnp
from hydra.conf import MISSING, dataclass
from jaxtyping import Array, Float
from loguru import logger

from mdpax.core.problem import Problem, ProblemConfig
from mdpax.utils.batch_processing import BatchProcessor
from mdpax.utils.logging import verbosity_to_loguru_level
from mdpax.utils.types import (
    BatchedStates,
)


@dataclass
class SolverConfig:
    """Base configuration for all MDP solvers.

    This serves as the base configuration class that all specific solver
    configurations should inherit from. It defines common parameters used
    across different solvers.
    """

    _target_: str = MISSING
    problem: ProblemConfig = MISSING

    # Solver parameters
    gamma: float = 0.99
    max_iter: int = 1000
    epsilon: float = 1e-3
    max_batch_size: int = 1024
    jax_double_precision: bool = True
    verbose: int = 2  # Default to INFO level


@dataclass
class SolverWithCheckpointConfig(SolverConfig):
    """Configuration for solvers with checkpointing."""

    checkpoint_dir: str | None = None
    checkpoint_frequency: int = 0
    max_checkpoints: int = 1
    enable_async_checkpointing: bool = True


@chex.dataclass(frozen=True)
class SolverInfo:
    """Base solver information.

    Contains common metadata needed by all solvers. Specific solvers
    can extend this with additional fields.

    Attributes:
        iteration: Current iteration count
    """

    iteration: int


@chex.dataclass(frozen=True)
class SolverState:
    """Base runtime state for all solvers.

    Contains the core state that must be maintained by all solvers.
    Specific solvers can extend the info field with solver-specific
    metadata.

    Attributes:
        values: Current value function [n_states]
        policy: Current policy (if computed) [n_states, action_dim]
        info: Solver metadata
    """

    values: Float[Array, "n_states"]
    policy: Float[Array, "n_states action_dim"] | None
    info: SolverInfo


class Solver(ABC):
    """Abstract base class for MDP solvers.

    Provides common functionality for solving MDPs using parallel processing
    across devices with batched state updates.

    Shape Requirements:
        Values and policies maintain consistent dimensionality:
        - Values: [n_states]
        - Policy: [n_states, action_dim]
        - Batched states: [n_devices, n_batches, batch_size, state_dim]

    Note:
        All array operations use JAX for efficient parallel processing.
        States are automatically batched and padded for device distribution.

    Args:
        problem: MDP problem to solve
        gamma: Discount factor in [0,1]
        max_iter: Maximum iterations
        epsilon: Convergence threshold
        max_batch_size: Maximum size of state batches
        jax_double_precision: Whether to use double precision
        verbose: Verbosity level (0-4)
    """

    def __init__(
        self,
        problem: Problem,
        gamma: float = 0.99,
        max_iter: int = 1000,
        epsilon: float = 1e-3,
        max_batch_size: int = 1024,
        jax_double_precision: bool = True,
        verbose: int = 2,
    ):
        if not isinstance(problem, Problem):
            raise TypeError("problem must be an instance of Problem")
        assert 0 <= gamma <= 1, "Discount factor must be in [0,1]"
        assert max_iter > 0, "Max iterations must be positive"
        assert epsilon > 0, "Epsilon must be positive"
        assert max_batch_size > 0, "Batch size must be positive"

        self.jax_double_precision = jax_double_precision
        if self.jax_double_precision:
            jax.config.update("jax_enable_x64", True)

        self.max_batch_size = max_batch_size
        self.problem = problem
        self.gamma = jnp.array(gamma)
        self.max_iter = max_iter
        self.epsilon = epsilon

        # Set initial verbosity
        self.set_verbosity(verbose)

        logger.info(f"Solver initialized with {problem.name} problem")
        logger.debug(f"Number of states: {problem.n_states}")
        logger.debug(f"Number of actions: {problem.n_actions}")
        logger.debug(f"Number of random events: {problem.n_random_events}")

        # Setup batch processing and solver-specific structures
        # and JIT compile functions (as part of pmap)
        self._setup()

    def set_verbosity(self, level: int | str) -> None:
        """Set the verbosity level for solver output.

        Args:
            level: Verbosity level, either as integer (0-4) or string
                  ('ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE')

        Integer levels map to:
            0: Minimal output (only errors)
            1: Show warnings and errors
            2: Show main progress (default)
            3: Show detailed progress
            4: Show everything
        """
        # Handle string input
        if isinstance(level, str):
            level = level.upper()
            valid_levels = {"ERROR": 0, "WARNING": 1, "INFO": 2, "DEBUG": 3, "TRACE": 4}
            if level not in valid_levels:
                raise ValueError(f"Invalid verbosity level: {level}")
            level = valid_levels[level]

        # Convert to loguru level
        loguru_level = verbosity_to_loguru_level(level)
        logger.remove()
        logger.add(sys.stderr, level=loguru_level)
        self.verbose = level

        logger.debug(f"Verbosity set to {level} ({loguru_level})")

    def _setup(self) -> None:
        """Setup batching and solver computations and JIT compile functions."""

        # Set up batch processing
        self.batch_processor = BatchProcessor(
            n_states=self.problem.n_states,
            state_dim=self.problem.state_space.shape[1],
            max_batch_size=self.max_batch_size,
        )

        self.batched_states: BatchedStates = self.batch_processor.prepare_batches(
            self.problem.state_space
        )

        # Initialize solver state
        self.values: Float[Array, "n_states"] | None = None
        self.policy: Float[Array, "n_states action_dim"] | None = None
        self.iteration: int = 0

        self._calculate_initial_value_scan_state_batches_pmap = jax.pmap(
            self._calculate_initial_value_scan_state_batches, in_axes=0
        )

        # Setup other solver-specific computations
        self._setup_solver()

        # Initialize values using solver-specific method
        self.values = self._initialize_values(self.batched_states)

    def _initialize_values(
        self, batched_states: BatchedStates
    ) -> Float[Array, "n_states"]:
        """Initialize value function using problem's initial value function.

        Uses pmap with batching to efficiently compute initial values for all states.

        Returns:
            Array of initial values for all states [n_states]
        """
        # Multi-device initialization using scan and pmap
        padded_batched_initial_values = (
            self._calculate_initial_value_scan_state_batches_pmap(batched_states)
        )

        initial_values = self._unbatch_results(padded_batched_initial_values)

        return initial_values

    def _calculate_initial_value_state_batch(
        self, carry, state_batch: Float[Array, "batch_size state_dim"]
    ) -> tuple[None, Float[Array, "batch_size"]]:
        """Calculate the updated value for a batch of states"""
        initial_values = jax.vmap(
            self.problem.initial_value,
        )(state_batch)
        return carry, initial_values

    def _calculate_initial_value_scan_state_batches(
        self,
        padded_batched_states: BatchedStates,
    ) -> Float[Array, "n_devices n_batches batch_size"]:
        """Calculate the updated value for multiple batches of states"""

        _, new_values_padded = jax.lax.scan(
            self._calculate_initial_value_state_batch,
            None,
            padded_batched_states,
        )
        return new_values_padded

    @abstractmethod
    def _setup_solver(self) -> None:
        """Setup solver-specific data structures and functions."""
        pass

    @abstractmethod
    def _iteration_step(self) -> tuple[Float[Array, "n_states"], float]:
        """Perform one iteration step.

        Returns:
            Tuple of (new values, convergence measure)
        """
        pass

    def solve(self) -> SolverState:
        """Run solver to convergence or max iterations.

        Returns:
            SolverState containing final values, policy, and solver info
        """
        while self.iteration < self.max_iter:
            new_values, conv = self._iteration_step()
            if conv < self.epsilon:
                break
            self.values = new_values
            self.iteration += 1
        return self.solver_state

    @property
    def solver_state(self) -> SolverState:
        """Get solver state for checkpointing."""
        return SolverState(
            values=self.values,
            policy=self.policy,
            info=SolverInfo(iteration=self.iteration),
        )

    def _unbatch_results(self, padded_array: jnp.ndarray) -> jnp.ndarray:
        """Reshape and remove padding from results."""
        return self.batch_processor.unbatch_results(padded_array)

    @property
    def n_devices(self) -> int:
        """Number of available devices."""
        return self.batch_processor.n_devices

    @property
    def batch_size(self) -> int:
        """Actual batch size being used."""
        return self.batch_processor.batch_size

    @property
    def n_pad(self) -> int:
        """Number of padding elements added."""
        return self.batch_processor.n_pad
