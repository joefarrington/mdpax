"""Base class for MDP solvers."""

import sys
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from hydra.conf import MISSING, dataclass
from loguru import logger

from mdpax.utils.logging import verbosity_to_loguru_level

from .problem import Problem, ProblemConfig


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
    batch_size: int = 1024
    jax_double_precision: bool = True
    verbose: int = 2  # Default to INFO level


@dataclass
class SolverWithCheckpointConfig(SolverConfig):
    """Configuration for solvers with checkpointing."""

    checkpoint_dir: Optional[str] = None
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

    values: chex.Array
    policy: Optional[chex.Array]
    info: SolverInfo


class Solver(ABC):
    """Abstract base class for MDP solvers.

    Provides common functionality for solving MDPs using parallel processing
    across devices with batched state updates.

    Args:
        problem: MDP problem to solve
        gamma: Discount factor in [0,1]
        max_iter: Maximum iterations
        epsilon: Convergence threshold
        batch_size: Size of state batches
        jax_double_precision: Whether to use double precision for JAX operations
        verbose: Verbosity level (0-4)
            0: Minimal output (only errors)
            1: Show warnings and errors
            2: Show main progress (default)
            3: Show detailed progress
            4: Show everything

    Attributes:
        problem: MDP problem being solved
        gamma: Discount factor
        max_iter: Maximum iterations
        epsilon: Convergence threshold
        batch_size: Size of state batches
        n_devices: Number of available devices
        values: Current value function [n_states]
        policy: Current policy (if computed) [n_states, action_dim]
        iteration: Current iteration count
        n_pad: Number of padding elements added to make sizes divide evenly
        batched_states: States prepared for batch processing
            Shape: [n_devices, n_batches, batch_size, state_dim]
            where:
            - n_devices is number of available devices (1 for single device)
            - n_batches is ceil(n_states / batch_size)
            - batch_size is min(batch_size, n_states) for single device
              or min(batch_size, max(64, n_states/n_devices)) for multiple devices
            - Padding is added if needed to make sizes divide evenly
    """

    def __init__(
        self,
        problem: Problem,
        gamma: float = 0.9,
        max_iter: int = 1000,
        epsilon: float = 1e-3,
        batch_size: int = 1024,
        jax_double_precision: bool = True,
        verbose: int = 2,
    ):
        if not isinstance(problem, Problem):
            raise TypeError("problem must be an instance of Problem")
        assert 0 <= gamma <= 1, "Discount factor must be in [0,1]"
        assert max_iter > 0, "Max iterations must be positive"
        assert epsilon > 0, "Epsilon must be positive"
        assert batch_size > 0, "Batch size must be positive"

        self.jax_double_precision = jax_double_precision
        if self.jax_double_precision:
            jax.config.update("jax_enable_x64", True)

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

        # Setup device information
        self.n_devices = len(jax.devices())

        # Calculate appropriate batch size
        if self.n_devices == 1:
            # Single device - clip to problem size and max batch
            self.batch_size = min(batch_size, problem.n_states)
        else:
            # Multiple devices - ensure even distribution
            states_per_device = problem.n_states // self.n_devices
            self.batch_size = min(
                batch_size,  # user provided/default max
                max(64, states_per_device),  # ensure minimum batch size
            )
        logger.debug(f"Number of devices: {self.n_devices}")

        # Initialize solver state
        self.values: Optional[jnp.ndarray] = None
        self.policy: Optional[jnp.ndarray] = None
        self.iteration: int = 0
        self.n_pad: int = 0  # Will be set in _setup_batch_processing
        self.batched_states: jnp.ndarray = (
            None  # Will be set in _setup_batch_processing
        )

        # Setup batch processing and solver-specific structures
        # and JIT compile functions (as part of pmap)
        self._setup()

        # Initialize values using solver-specific method
        self.values = self._initialize_values(self.batched_states)

    def set_verbosity(self, level: Union[int, str]) -> None:
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
        """Setup solver computations and JIT compile functions."""
        self._setup_batch_processing()

        self._calculate_initial_value_scan_state_batches_pmap = jax.pmap(
            self._calculate_initial_value_scan_state_batches, in_axes=0
        )

        # Setup other solver-specific computations
        self._setup_solver()

    def _initialize_values(self, batched_states: jnp.ndarray) -> jnp.ndarray:
        """Initialize value function using problem's initial value function.

        Uses pmap with batching to efficiently compute initial values for all states.

        Returns:
            Array of initial values for all states [n_states]
        """
        # Multi-device initialization using scan and pmap
        padded_batched_initial_values = (
            self._calculate_initial_value_scan_state_batches_pmap(batched_states)
        )
        padded_initial_values = jnp.reshape(padded_batched_initial_values, (-1,))

        initial_values = self._unpad_results(padded_initial_values)

        return initial_values

    def _calculate_initial_value_state_batch(
        self, carry, state_batch: chex.Array
    ) -> Tuple[None, chex.Array]:
        """Calculate the updated value for a batch of states"""
        initial_values = jax.vmap(
            self.problem.initial_value,
        )(state_batch)
        return carry, initial_values

    def _calculate_initial_value_scan_state_batches(
        self,
        padded_batched_states: chex.Array,
    ) -> chex.Array:
        """Calculate the updated value for multiple batches of states, using
        jax.lax.scan to loop over batches of states."""

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
    def _iteration_step(self) -> Tuple[jnp.ndarray, float]:
        """Perform one iteration step.

        Returns:
            Tuple of (new values, convergence measure)
        """
        pass

    def solve(self) -> SolverState:
        """Run solver to convergence.

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

    def _setup_batch_processing(self) -> None:
        """Setup batch processing based on problem size and devices.

        Always reshapes states to (n_devices, n_batches, batch_size, state_dim) where:
        - n_devices is number of available devices (1 for single device)
        - n_batches is 1 if problem size <= batch_size
        - Padding is added if needed to make sizes divide evenly

        Uses pmap for all cases for consistency, which handles single-device
        case automatically.
        """
        states = self.problem.state_space
        state_dim = states.shape[1]
        n_states = len(states)

        # Determine batch dimensions
        if n_states <= self.batch_size:
            # Small problem - single batch
            n_batches = 1
            actual_batch_size = n_states
        else:
            # Multiple batches needed
            n_batches = (n_states + self.batch_size - 1) // self.batch_size
            actual_batch_size = self.batch_size

        # Calculate padding needed
        total_size = self.n_devices * n_batches * actual_batch_size
        self.n_pad = total_size - n_states

        # Pad if needed
        if self.n_pad > 0:
            states = jnp.vstack(
                [states, jnp.zeros((self.n_pad, state_dim), dtype=states.dtype)]
            )

        # Reshape to standard format: (devices, batches, batch_size, state_dim)
        self.batched_states = states.reshape(
            self.n_devices, n_batches, actual_batch_size, state_dim
        )

    def _unpad_results(self, padded_array: jnp.ndarray) -> jnp.ndarray:
        """Remove padding from results if needed."""
        if self.n_pad > 0:
            return padded_array[: -self.n_pad]
        return padded_array
