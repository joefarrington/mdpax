"""Base class for MDP solvers."""

from abc import ABC, abstractmethod
from typing import Tuple

import jax
import jax.numpy as jnp
from loguru import logger

from mdpax.core.problem import Problem


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

    Attributes:
        problem: MDP problem being solved
        gamma: Discount factor
        max_iter: Maximum iterations
        epsilon: Convergence threshold
        batch_size: Size of state batches
        n_devices: Number of available devices
        values: Current value function
        policy: Current policy (if computed)
        iteration: Current iteration count
        n_pad: Number of padding elements
        batched_states: States prepared for batch processing
    """

    def __init__(
        self,
        problem: Problem,
        gamma: float = 0.9,
        max_iter: int = 1000,
        epsilon: float = 1e-3,
        batch_size: int = 1024,
        jax_double_precision: bool = True,
    ):
        if not isinstance(problem, Problem):
            raise TypeError("problem must be an instance of Problem")
        assert 0 <= gamma <= 1, "Discount factor must be in [0,1]"
        assert max_iter > 0, "Max iterations must be positive"
        assert epsilon > 0, "Epsilon must be positive"
        assert batch_size > 0, "Batch size must be positive"

        if jax_double_precision:
            jax.config.update("jax_enable_x64", True)

        self.problem = problem
        self.gamma = gamma
        self.max_iter = max_iter
        self.epsilon = epsilon

        logger.info(f"Solver initialized with {problem.name} problem")
        logger.info(f"Number of states: {problem.n_states}")
        logger.info(f"Number of actions: {problem.n_actions}")
        logger.info(f"Number of random events: {problem.n_random_events}")

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
        logger.info(f"Number of devices: {self.n_devices}")

        # Initialize solver state
        self.values = None
        self.policy = None
        self.iteration = 0

        # Setup processing based on problem size and devices
        self._setup_processing()

        # Setup solver-specific structures and JIT compile functions
        self._setup()

        # Initialize values using solver-specific method
        self.values = self._initialize_values()

    def _setup(self) -> None:
        """Setup solver computations and JIT compile functions."""
        # JIT compile initial value computation for single device
        self._compute_initial_values = jax.jit(jax.vmap(self.problem.initial_value))

        # Setup multi-device initial value computation
        if self.n_devices > 1:
            self._compute_initial_values_pmap = jax.pmap(
                lambda x: jax.vmap(self.problem.initial_value)(x),
                in_axes=(0,),  # batched_states
                axis_name="device",
            )
            # Setup scan for initialization
            self._scan_initial_values = jax.pmap(
                lambda states: jax.vmap(self.problem.initial_value)(states),
                in_axes=(0,),  # batched_states
                axis_name="device",
            )
        else:
            # Setup scan for single device
            batch_fn = self._compute_initial_values

            def scan_fn(_, batch):
                return (None, batch_fn(batch))

            self._scan_initial_values = jax.jit(
                lambda states: jax.lax.scan(scan_fn, None, states)[1]
            )

        # Setup other solver-specific computations
        self._setup_solver()

    def _initialize_values(self) -> jnp.ndarray:
        """Initialize value function using problem's initial value function.

        Uses the same batching and multi-device infrastructure as value updates
        to efficiently compute initial values for all states.

        Returns:
            Array of initial values for all states [n_states]
        """
        if self.n_devices > 1:
            # Multi-device initialization using scan
            device_values = self._scan_initial_values(self.batched_states)
            values = jnp.reshape(device_values, (-1,))
        else:
            # Single device initialization using scan
            values = self._scan_initial_values(self.batched_states)
            values = jnp.reshape(values, (-1,))

        # Remove padding if needed
        if self.n_pad > 0:
            values = values[: -self.n_pad]

        return values

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

    @abstractmethod
    def _batch_value_calculation(
        self, states: jnp.ndarray, values: jnp.ndarray, *args
    ) -> jnp.ndarray:
        """Process a batch of states."""
        pass

    @abstractmethod
    def _scan_batches(
        self,
        carry: Tuple[jnp.ndarray, float],  # (values, gamma)
        batched_states: jnp.ndarray,
    ) -> jnp.ndarray:
        """Process batches of states (for multi-device case)."""
        pass

    def solve(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Run solver to convergence.

        Returns:
            Tuple of (optimal values, optimal policy)
        """
        while self.iteration < self.max_iter:
            new_values, conv = self._iteration_step()
            if conv < self.epsilon:
                break
            self.values = new_values
            self.iteration += 1
        return self.values, self.policy

    def _setup_processing(self) -> None:
        """Setup batch processing based on problem size and devices."""
        if self.n_devices == 1 and self.problem.n_states <= self.batch_size:
            # Small problem on single device - no batching needed
            self.batched_states = self.problem.state_space.reshape(
                1, -1, self.problem.state_space.shape[1]
            )
            self.n_pad = 0
            self._process_batches = self._setup_single_small()
        elif self.n_devices == 1:
            # Larger problem on single device - batch but no device padding
            self.batched_states, self.n_pad = self._prepare_single_device_batches()
            self._process_batches = self._setup_single_device()
        else:
            # Multiple devices - batch for parallel processing
            self.batched_states, self.n_pad = self._prepare_multi_device_batches()
            self._process_batches = self._setup_multi_device()

    def _prepare_single_device_batches(self) -> Tuple[jnp.ndarray, int]:
        """Prepare batches for single device case."""
        states = self.problem.state_space
        n_batches = (len(states) + self.batch_size - 1) // self.batch_size
        total_size = n_batches * self.batch_size
        n_pad = total_size - len(states)

        # Pad if needed
        if n_pad > 0:
            states = jnp.vstack(
                [states, jnp.zeros((n_pad, states.shape[1]), dtype=states.dtype)]
            )

        # Reshape into batches
        return states.reshape(-1, self.batch_size, states.shape[1]), n_pad

    def _prepare_multi_device_batches(self) -> Tuple[jnp.ndarray, int]:
        """Prepare batches for multiple devices."""
        states = self.problem.state_space
        n_pad = (self.n_devices * self.batch_size) - (
            len(states) % (self.n_devices * self.batch_size)
        )

        # Pad states
        padded_states = jnp.vstack(
            [states, jnp.zeros((n_pad, states.shape[1]), dtype=states.dtype)]
        )

        # Reshape for devices and batches
        return (
            padded_states.reshape(self.n_devices, -1, self.batch_size, states.shape[1]),
            n_pad,
        )

    def _setup_single_small(self):
        """Setup processing for small problems (no batching needed)."""
        batch_fn = jax.jit(self._batch_value_calculation)
        return lambda carry, states: batch_fn(states[0], *carry)

    def _setup_single_device(self):
        """Setup processing for single device."""
        batch_fn = jax.jit(self._batch_value_calculation)

        def process_batches(carry, states):
            def scan_fn(_, batch):
                return (None, batch_fn(batch, *carry))

            _, new_values = jax.lax.scan(scan_fn, None, states)
            return new_values

        return jax.jit(process_batches)

    def _setup_multi_device(self):
        """Setup processing for multiple devices."""
        return jax.pmap(
            self._scan_batches,
            in_axes=((None, None), 0),  # (values, gamma), batched_states
        )

    def _unpad_results(self, padded_array: jnp.ndarray) -> jnp.ndarray:
        """Remove padding from results array."""
        return padded_array[: -self.n_pad] if self.n_pad > 0 else padded_array
