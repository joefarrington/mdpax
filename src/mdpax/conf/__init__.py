"""Configuration registration for mdpax.

This module handles registration of all configuration schemas with Hydra's
ConfigStore. This enables type-safe configuration of problems and solvers.
"""

from hydra.core.config_store import ConfigStore

from ..core.problem import ProblemConfig
from ..core.solver import SolverConfig
from ..solvers.periodic_value_iteration import PeriodicValueIterationConfig
from ..solvers.relative_value_iteration import RelativeValueIterationConfig
from ..solvers.value_iteration import ValueIterationConfig

# Get the global config store instance
cs = ConfigStore.instance()

# Register base configs
cs.store(group="problem", name="base", node=ProblemConfig)
cs.store(group="solver", name="base", node=SolverConfig)

# Register solver configs
cs.store(group="solver", name="value_iteration", node=ValueIterationConfig)
cs.store(
    group="solver", name="relative_value_iteration", node=RelativeValueIterationConfig
)
cs.store(
    group="solver", name="periodic_value_iteration", node=PeriodicValueIterationConfig
)
