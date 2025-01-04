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
    BatchedResults,
    BatchedStates,
    Policy,
    ResultsBatch,
    StateBatch,
    ValueFunction,
)


@dataclass
class SolverConfig:
    """Base configuration for all MDP solvers.

    This serves as the base configuration class that all specific solver
    configurations should inherit from. It defines common parameters used
    across different solvers.
    """

    _target_: str = MISSING
    gamma: float = MISSING
    problem: ProblemConfig = MISSING
    max_batch_size: int = MISSING
    jax_double_precision: bool = MISSING
    verbose: int = MISSING


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

    values: ValueFunction | None
    policy: Policy | None
    info: SolverInfo


class Solver(ABC):
    """Abstract base class for MDP solvers.

    Provides common functionality for solving MDPs using parallel processing
    across devices with batched state updates. Subclasses must implement
    the core solution algorithm while inheriting the parallel processing
    and batching infrastructure.

    Required Implementations:
        _setup_solver() -> None:
            Setup solver-specific data structures and functions.
            Called after base initialization.

        _iteration_step() -> tuple[ValueFunction, float]:
            Perform one iteration of the solution algorithm.
            Returns (new_values, convergence_measure).

    Optional Implementations:
        _initialize_values(batched_states: BatchedStates) -> ValueFunction:
            Initialize value function. Default uses problem's initial_value.

    Shape Requirements:
        - Values: [n_states]
        - Policy: [n_states, action_dim]
        - Batched states: [n_devices, n_batches, batch_size, state_dim]
        - Batched results: [n_devices, n_batches, batch_size]

    Args:
        problem: MDP problem to solve
        gamma: Discount factor in [0,1]
        epsilon: Convergence threshold
        max_batch_size: Maximum states to process in parallel on each device
        jax_double_precision: Whether to use float64 precision
        verbose: Logging verbosity level (0-4)

    Attributes:
        problem: Problem instance being solved
        gamma: Discount factor
        epsilon: Convergence threshold
        max_batch_size: Maximum batch size for parallel processing
        values: Current value function [n_states] or None
        policy: Current policy [n_states, action_dim] or None
        iteration: Current iteration count
        batch_processor: Utility for handling batched computations
        verbose: Current verbosity level
        solver_state: Current solver state containing values, policy, and info
        n_devices: Number of available JAX devices for parallel processing
        batch_size: Actual batch size being used (may be less than max_batch_size)
        n_pad: Number of padding elements added to make batches fit devices

    Note:
        - All array operations use JAX for efficient parallel processing
        - States are automatically batched and padded for device distribution
        - Subclasses should use jax.jit and jax.pmap for performance
    """

    def __init__(
        self,
        problem: Problem,
        gamma: float = 0.99,
        epsilon: float = 1e-3,
        max_batch_size: int = 1024,
        jax_double_precision: bool = True,
        verbose: int = 2,
    ):
        if not isinstance(problem, Problem):
            raise TypeError("problem must be an instance of Problem")
        assert 0 <= gamma <= 1, "Discount factor must be in [0,1]"
        assert epsilon > 0, "Epsilon must be positive"
        assert max_batch_size > 0, "Batch size must be positive"

        self.jax_double_precision = jax_double_precision
        if self.jax_double_precision:
            jax.config.update("jax_enable_x64", True)

        self.max_batch_size = max_batch_size
        self.problem = problem
        self.gamma = jnp.array(gamma)
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
            - 0: Minimal output (only errors)
            - 1: Show warnings and errors
            - 2: Show main progress (default)
            - 3: Show detailed progress
            - 4: Show everything
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
        self.values: ValueFunction | None = None
        self.policy: Policy | None = None
        self.iteration: int = 0

        self._calculate_initial_value_scan_state_batches_pmap = jax.pmap(
            self._calculate_initial_value_scan_state_batches, in_axes=0
        )

        # Setup other solver-specific computations
        self._setup_solver()

        # Initialize values using solver-specific method
        self.values = self._initialize_values(self.batched_states)

    def _initialize_values(self, batched_states: BatchedStates) -> ValueFunction:
        """Initialize value function using problem's initial value function."""
        # Multi-device initialization using scan and pmap
        padded_batched_initial_values = (
            self._calculate_initial_value_scan_state_batches_pmap(batched_states)
        )

        initial_values = self._unbatch_results(padded_batched_initial_values)

        return initial_values

    def _calculate_initial_value_state_batch(
        self, carry, state_batch: StateBatch
    ) -> tuple[None, ResultsBatch]:
        """Calculate the updated value for a batch of states"""
        initial_values = jax.vmap(
            self.problem.initial_value,
        )(state_batch)
        return carry, initial_values

    def _calculate_initial_value_scan_state_batches(
        self,
        padded_batched_states: BatchedStates,
    ) -> BatchedResults:
        """Calculate the updated value for multiple batches of states"""

        _, padded_batched_initial_values = jax.lax.scan(
            self._calculate_initial_value_state_batch,
            None,
            padded_batched_states,
        )
        return padded_batched_initial_values

    @abstractmethod
    def _setup_solver(self) -> None:
        """Setup solver-specific data structures and functions.

        Returns:
            None
        """
        pass

    @abstractmethod
    def _iteration_step(self) -> tuple[ValueFunction, float]:
        """Perform one iteration of the solution algorithm.

        Returns:
            Tuple of (new values, convergence measure) where new values has shape
            [n_states]
        """
        pass

    def solve(self, max_iterations: int = 2000) -> SolverState:
        """Run solver to convergence or max iterations.

        Args:
            max_iterations: Maximum number of iterations to run

        Returns:
            SolverState containing final values [n_states], optimal policy [n_states, action_dim],
            and SolverInfo including iteration count
        """
        for _ in range(max_iterations):
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

    def _unbatch_results(
        self,
        padded_batched_results: Float[Array, "n_devices n_batches batch_size *dims"],
    ) -> Float[Array, "n_states *dims"]:
        """Remove padding from batched results and combine across devices."""
        return self.batch_processor.unbatch_results(padded_batched_results)

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
