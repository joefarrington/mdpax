# -*- coding: utf-8 -*-
"""Markov Decision Process (MDP) Toolbox: ``example`` module
=========================================================

The ``example`` module provides functions to generate valid MDP transition and
reward matrices using JAX for accelerated computation.

Available functions
-------------------

:func:`~mdptoolbox.example.forest`
    A simple forest management example using JAX
"""

import warnings

import jax.numpy as jnp
from jax.experimental import sparse as jsp

from mdpax.problems.forest import Forest


def forest(
    S: int = 3, r1: float = 4, r2: float = 2, p: float = 0.1, is_sparse: bool = False
) -> tuple[jnp.ndarray | jsp.BCOO, jnp.ndarray]:
    """Generate a MDP example based on a simple forest management scenario.

    This is a JAX implementation of the forest management MDP problem.
    Note: Sparse matrix support uses JAX's experimental sparse module.

    Parameters
    ---------
    S : int, optional
        The number of states, which should be an integer greater than 1.
        Default: 3.
    r1 : float, optional
        The reward when the forest is in its oldest state and action 'Wait' is
        performed. Default: 4.
    r2 : float, optional
        The reward when the forest is in its oldest state and action 'Cut' is
        performed. Default: 2.
    p : float, optional
        The probability of wild fire occurence, in the range ]0, 1[. Default:
        0.1.
    is_sparse : bool, optional
        If True, returns transition matrices in JAX experimental sparse format.
        Default: False.

    Returns
    -------
    out : tuple
        ``out[0]`` contains the transition probability matrix P and ``out[1]``
        contains the reward matrix R. P has shape (A, S, S) in both dense and
        sparse formats, though the sparse version uses JAX's BCOO format.
        R always has shape (S, A).
    """
    assert S > 1, "The number of states S must be greater than 1."
    assert (r1 > 0) and (r2 > 0), "The rewards must be non-negative."
    assert 0 <= p <= 1, "The probability p must be in [0; 1]."

    problem = Forest(S, r1, r2, p)
    P, R = problem.build_matrices()

    if is_sparse:
        warnings.warn(
            "Using experimental JAX sparse matrix support. API may change in future \
            versions.\
            See https://jax.readthedocs.io/en/latest/jax.experimental.sparse.html \
            for more information.",
            UserWarning,
        )
        P = jsp.BCOO.fromdense(P)

    return P, R
