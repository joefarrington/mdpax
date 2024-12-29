"""Type definitions for MDP components."""

from typing import TypeAlias

from jaxtyping import Array, Float

# State types
StateVector: TypeAlias = Float[Array, "state_dim"]
StateSpace: TypeAlias = Float[Array, "n_states state_dim"]
StateBatch: TypeAlias = Float[Array, "batch_size state_dim"]
BatchedStates: TypeAlias = Float[
    Array, "n_devices n_batches_per_device batch_size state_dim"
]

# Action types
ActionVector: TypeAlias = Float[Array, "action_dim"]
ActionSpace: TypeAlias = Float[Array, "n_actions action_dim"]

# Random event types
RandomEventVector: TypeAlias = Float[Array, "event_dim"]
RandomEventSpace: TypeAlias = Float[Array, "n_events event_dim"]

# Other types
Reward: TypeAlias = float
