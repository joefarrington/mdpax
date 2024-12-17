"""Markov Decision Process (MDP) Toolbox: ``mdp`` module
=====================================================

JAX implementation of discrete-time Markov Decision Processes.
"""
import jax
jax.config.update("jax_enable_x64", True)

import math as _math
import time as _time
from typing import Optional, TypeAlias
import jax.numpy as jnp
from jax import vmap, jit, lax
import jax.scipy.sparse as jsp
import numpy as np
from functools import partial

# Type aliases
JaxInt: TypeAlias = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32

# Constants
_MSG_STOP_MAX_ITER = "Iterating stopped due to maximum number of iterations condition."
_MSG_STOP_EPSILON_OPTIMAL_POLICY = "Iterating stopped, epsilon-optimal policy found."
_MSG_STOP_EPSILON_OPTIMAL_VALUE = "Iterating stopped, epsilon-optimal value function found."
_MSG_STOP_UNCHANGING_POLICY = "Iterating stopped, unchanging policy found."

def _compute_dimensions(transition: jnp.ndarray | tuple[jnp.ndarray, ...]) -> tuple[int, int]:
    """Compute state and action dimensions from transition matrices."""
    if isinstance(transition, tuple):
        A = len(transition)
        S = transition[0].shape[0]
    else:
        A = transition.shape[0]
        S = transition.shape[1]
    return S, A

def _print_verbosity(iteration: int, variation: float | int) -> None:
    """Print iteration information in verbose mode."""
    if isinstance(variation, float):
        print(f"{iteration:>10}{variation:>12.6f}")
    else:
        print(f"{iteration:>10}{variation:>12}")

@jit
def get_span(array: jnp.ndarray) -> jnp.ndarray:
    """Calculate the span of an array (max - min)."""
    return jnp.max(array) - jnp.min(array)

class MDP:
    """A Markov Decision Problem implemented with JAX.

    Let ``S`` = the number of states, and ``A`` = the number of actions.

    Parameters
    ----------
    transitions : array or tuple
        Transition probability matrices. Can be provided as:
        - A JAX/numpy array with shape (A, S, S)
        - A tuple of length A containing matrices of shape (S, S)
        Each action's transition matrix must be indexable as transitions[a]
        where a ∈ {0, 1...A-1}, returning an S × S array.
    reward : array or tuple
        Reward matrices or vectors. Can be provided as:
        - A JAX/numpy array with shape (S, A), (S,) or (A, S, S)
        - A tuple of arrays, each of shape (S,)
        Each reward must be indexable as reward[a] for action a.
    discount : float
        Discount factor. Must be in range (0, 1]. If discount=1, convergence
        is not guaranteed and a warning will be displayed. Can be None for
        algorithms that don't use discounting.
    epsilon : float
        Stopping criterion. The maximum change in the value function at each
        iteration is compared against epsilon. Can be None for algorithms
        that don't use epsilon-optimal stopping.
    max_iter : int
        Maximum number of iterations. Must be greater than 0 if specified.
        Can be None for algorithms that don't use iteration limits.
    skip_check : bool
        If True, skips input validation checks. Default: False.

    Attributes
    ----------
    P : jax.numpy.ndarray
        Transition probability matrices with shape (A, S, S).
    R : jax.numpy.ndarray
        Reward vectors with shape (A, S).
    V : jax.numpy.ndarray
        The optimal value function with shape (S,).
    discount : float
        The discount rate on future rewards.
    max_iter : int
        The maximum number of iterations.
    policy : jax.numpy.ndarray
        The optimal policy with shape (S,).
    time : float
        The time used to converge to the optimal policy.
    verbose : bool
        Whether verbose output should be displayed.

    Methods
    -------
    run()
        Implemented in child classes as the main algorithm loop.
    set_silent()
        Turn verbosity off.
    set_verbose()
        Turn verbosity on.

    Notes
    -----
    This implementation uses JAX for accelerated computation, including
    automatic vectorization and JIT compilation where appropriate.
    
    Note that using verbose mode disables JIT compilation and efficient
    JAX loops, significantly impacting performance. Use verbose mode only
    for debugging and development.
    """
    
    def __init__(
        self,
        transitions: jnp.ndarray | tuple[jnp.ndarray, ...],
        reward: jnp.ndarray | tuple[jnp.ndarray, ...],
        discount: Optional[float] = None,
        epsilon: Optional[float] = None,
        max_iter: Optional[int] = None,
        skip_check: bool = False
    ) -> None:
        if discount is not None:
            self.discount: float = float(discount)
            assert 0.0 < self.discount <= 1.0, "Discount rate must be in ]0; 1]"
            if self.discount == 1:
                print("WARNING: With no discount, convergence cannot be assumed.")

        if max_iter is not None:
            self.max_iter: int = int(max_iter)
            assert self.max_iter > 0, "Maximum iterations must be greater than 0."

        if epsilon is not None:
            self.epsilon: float = float(epsilon)
            assert self.epsilon > 0, "Epsilon must be greater than 0."

        # Compute dimensions and convert inputs to JAX arrays
        self.S: int
        self.A: int
        self.S, self.A = _compute_dimensions(transitions)
        self.P: jnp.ndarray = self._compute_transition(transitions)
        self.R: jnp.ndarray = self._compute_reward(reward, transitions)

        self.verbose: bool = False
        self.time: Optional[float] = None
        self.iter: int = 0
        self.V: Optional[jnp.ndarray] = None
        self.policy: Optional[jnp.ndarray] = None

    @partial(jit, static_argnums=(0,))
    def _bellman_operator(
        self, 
        V: Optional[jnp.ndarray] = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the Bellman operator to the value function."""
        if V is None:
            V = self.V
            
        # Vectorize Q-value computation across actions
        def q_value_for_action(P_a: jnp.ndarray, R_a: jnp.ndarray) -> jnp.ndarray:
            discounted = self.discount * jnp.dot(P_a, V)
            result = R_a + discounted
            if self.verbose:
                print(f"R: {R_a}, discounted: {discounted}, result: {result}")
            return result
            
        Q: jnp.ndarray = vmap(q_value_for_action)(self.P, self.R)
        
        if self.verbose:
            print(f"Q matrix: {Q}")
            print(f"Discount: {self.discount}")
        
        return jnp.argmax(Q, axis=0), jnp.max(Q, axis=0)

    def _compute_transition(
        self, 
        transition: jnp.ndarray | tuple[jnp.ndarray, ...]
    ) -> jnp.ndarray:
        """Convert transition matrices to JAX array."""
        if isinstance(transition, tuple):
            return jnp.stack([jnp.array(t) for t in transition])
        return jnp.array(transition)

    def _compute_reward(
        self, 
        reward: jnp.ndarray | tuple[jnp.ndarray, ...],
        transition: jnp.ndarray | tuple[jnp.ndarray, ...]
    ) -> jnp.ndarray:
        """Convert reward matrices to JAX array."""
        if isinstance(reward, (tuple, list)):
            return jnp.stack([jnp.array(r).reshape(self.S) for r in reward])
        
        reward = jnp.array(reward)
        if reward.ndim == 1:
            return jnp.tile(reward.reshape(self.S), (self.A, 1))
        elif reward.ndim == 2:
            return reward.T
        else:
            # Handle 3D reward matrices
            return jnp.einsum('ast,ast->as', reward, transition)

    def run(self):
        """Raises error because child classes should implement this function."""
        raise NotImplementedError("You should create a run() method.")

    def _start_run(self):
        if self.verbose:
            _print_verbosity('Iteration', 'Variation')
        self.time = _time.time()

    def _end_run(self):
        self.V = jnp.array(self.V)
        self.policy = jnp.array(self.policy)
        self.time = _time.time() - self.time

    def set_silent(self):
        """Set the MDP algorithm to silent mode."""
        self.verbose = False

    def set_verbose(self):
        """Set the MDP algorithm to verbose mode."""
        self.verbose = True

class ValueIteration(MDP):
    """A discounted MDP solved using the value iteration algorithm with JAX.

    Description
    -----------
    ValueIteration applies the value iteration algorithm to solve a
    discounted MDP using JAX for accelerated computation. The algorithm
    solves Bellman's equation iteratively, using JIT compilation and
    automatic vectorization for improved performance.

    Parameters
    ----------
    transitions : array or tuple
        Transition probability matrices. See MDP class documentation.
    reward : array or tuple
        Reward matrices or vectors. See MDP class documentation.
    discount : float
        Discount factor. See MDP class documentation.
    epsilon : float, optional
        Stopping criterion. See MDP class documentation. Default: 0.01.
    max_iter : int, optional
        Maximum number of iterations. If discount < 1, a bound is computed.
        Otherwise, defaults to 1000.
    initial_value : array or float, optional
        The starting value function. Default: 0.
    skip_check : bool
        If True, skips input validation. Default: False.

    Attributes
    ----------
    V : jax.numpy.ndarray
        The optimal value function.
    policy : jax.numpy.ndarray
        The optimal policy function. Each element is an integer corresponding
        to an action which maximizes the value function in that state.
    iter : int
        The number of iterations taken to complete the computation.
    time : float
        The amount of CPU time used to run the algorithm.

    Notes
    -----
    The implementation uses JAX's lax.while_loop for the main iteration loop
    when running in non-verbose mode, allowing for JIT compilation of the
    entire computation. In verbose mode, it falls back to a Python-level loop
    to allow for progress reporting, which significantly reduces performance.
    Use verbose mode only for debugging and development.

    The algorithm stops when either:
    - An epsilon-optimal policy is found
    - The maximum number of iterations is reached

    Examples
    --------
    >>> import mdpax.mdptoolbox.example as example
    >>> P, R = example.forest()
    >>> vi = ValueIteration(P, R, 0.96)
    >>> vi.run()
    >>> print(vi.policy)  # Optimal policy
    >>> print(vi.V)  # Value function
    """
    
    def __init__(
        self,
        transitions: jnp.ndarray | tuple[jnp.ndarray, ...],
        reward: jnp.ndarray | tuple[jnp.ndarray, ...],
        discount: float,
        epsilon: float = 0.01,
        max_iter: int = 1000,
        initial_value: float | jnp.ndarray = 0,
        skip_check: bool = False
    ) -> None:
        super().__init__(transitions, reward, discount, epsilon, max_iter, skip_check)
        
        # Initialize value function with explicit type
        self.V: jnp.ndarray = (jnp.zeros(self.S) if initial_value == 0 
                              else jnp.array(initial_value).reshape(self.S))
        
        # Set convergence threshold
        self.thresh: float = (
            epsilon * (1 - self.discount) / self.discount 
            if self.discount < 1 
            else epsilon
        )
        
        if self.discount < 1:
            self._bound_iter(epsilon)

    def _bound_iter(self, epsilon: float) -> None:
        """Compute bound for number of iterations needed for convergence."""
        def min_probs_for_state(s: int) -> jnp.ndarray:
            return jnp.min(self.P[:, :, s])
        
        h: jnp.ndarray = vmap(min_probs_for_state)(jnp.arange(self.S))
        k: float = 1 - jnp.sum(h)
        
        # Compute initial span
        _, value = self._bellman_operator()
        span: float = get_span(value - self.V)
        
        # Compute max iterations bound
        max_iter: float = (_math.log((epsilon * (1 - self.discount) / self.discount) / span) / 
                          _math.log(self.discount * k))
        
        self.max_iter = int(_math.ceil(max_iter))

    @partial(jit, static_argnums=(0,))
    def _value_iteration_step(
        self,
        V: jnp.ndarray,
        i: int
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, bool]:
        """Single step of value iteration."""
        policy, new_V = self._bellman_operator(V)
        variation = get_span(new_V - V)
        done = jnp.logical_or(
            variation < self.thresh,
            i >= self.max_iter - 1
        )
        return new_V, policy, variation, done

    def run(self) -> None:
        """Run the value iteration algorithm."""
        self._start_run()
        
        if self.verbose:
            # Use Python-level loop when verbose output is needed
            V: jnp.ndarray = self.V
            variation: float = float('inf')
            policy: jnp.ndarray = jnp.zeros(self.S, dtype=JaxInt)
            
            while not (variation < self.thresh or self.iter >= self.max_iter):
                new_V, policy, variation, _done = self._value_iteration_step(V, self.iter)
                _print_verbosity(self.iter, float(variation))
                V = new_V
                self.iter += 1
                
            self.V = V
            self.policy = policy
            
            if variation < self.thresh:
                print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
            else:
                print(_MSG_STOP_MAX_ITER)
        else:
            def cond_fun(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, bool, int]) -> bool:
                _, _, _, done, _ = state
                return jnp.logical_not(done)
                
            def body_fun(
                state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, bool, int]
            ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, bool, int]:
                V: jnp.ndarray
                _: jnp.ndarray
                iter_count: int
                V, _, _, _, iter_count = state
                new_V, policy, variation, done = self._value_iteration_step(V, iter_count)
                return new_V, policy, variation, done, iter_count + 1
            
            # Initialize state for the loop
            init_state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, bool, int] = (
                self.V, 
                jnp.zeros(self.S, dtype=JaxInt), 
                jnp.array(float('inf')), 
                False,
                0  # Initial iteration count
            )
            
            # Run the iteration
            V: jnp.ndarray
            policy: jnp.ndarray
            _: jnp.ndarray
            _2: bool
            iter_count: int
            V, policy, _, _2, iter_count = lax.while_loop(
                cond_fun,
                body_fun,
                init_state
            )
            self.V = V
            self.policy = policy
            self.iter = int(iter_count)  # Set iteration count after loop
        
        self._end_run()