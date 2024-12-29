"""Type definitions for MDP components."""

from typing import TypeAlias

from jaxtyping import Array, Float

# State types
StateVector: TypeAlias = Float[Array, "state_dim"]
StateBatch: TypeAlias = Float[Array, "batch_state_dim"]
StateSpace: TypeAlias = Float[Array, "n_states state_dim"]

# Action types
ActionVector: TypeAlias = Float[Array, "action_dim"]
ActionBatch: TypeAlias = Float[Array, "batch_action_dim"]
ActionSpace: TypeAlias = Float[Array, "n_actions action_dim"]

# Random event types
RandomEventVector: TypeAlias = Float[Array, "event_dim"]
RandomEventSpace: TypeAlias = Float[Array, "n_events event_dim"]

# Other types
Reward: TypeAlias = float
