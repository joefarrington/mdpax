API Reference
=============

Core
----

Problem
~~~~~~~

.. autoclass:: mdpax.core.problem.Problem
   :members:
   :private-members: _construct_state_space, _construct_action_space, _construct_random_event_space, _setup_before_space_construction, _setup_after_space_construction
   :special-members: __init__
   :noindex:

.. autoclass:: mdpax.core.problem.ProblemConfig
   :members:
   :noindex:

Solver
~~~~~~

.. autoclass:: mdpax.core.solver.Solver
   :members:
   :private-members: _setup_solver, _iteration_step, _initialize_values
   :special-members: __init__
   :noindex:

.. autoclass:: mdpax.core.solver.SolverConfig
   :members:
   :noindex:

.. autoclass:: mdpax.core.solver.SolverState
   :members:
   :noindex:

.. autoclass:: mdpax.core.solver.SolverInfo
   :members:
   :noindex:

Solvers
-------

.. autoclass:: mdpax.solvers.value_iteration.ValueIteration
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mdpax.solvers.relative_value_iteration.RelativeValueIteration
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mdpax.solvers.periodic_value_iteration.PeriodicValueIteration
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

Problems
--------

Basic Problems
~~~~~~~~~~~~~~

.. autoclass:: mdpax.problems.forest.Forest
   :members:
   :undoc-members:
   :show-inheritance:


Perishable Inventory Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: mdpax.problems.perishable_inventory.de_moor_single_product.DeMoorSingleProductPerishable
   :members:
   :undoc-members:
   :show-inheritance: 

.. autoclass:: mdpax.problems.perishable_inventory.hendrix_two_product.HendrixTwoProductPerishable
   :members:
   :undoc-members:
   :show-inheritance: 

.. autoclass:: mdpax.problems.perishable_inventory.mirjalili_platelet.MirjaliliPlateletPerishable
   :members:
   :undoc-members:
   :show-inheritance: 

Utils
-----

Batch Processing
~~~~~~~~~~~~~~~~

.. autoclass:: mdpax.utils.batch_processing.BatchProcessor
   :members:
   :undoc-members:
   :show-inheritance: 

Checkpointing
~~~~~~~~~~~~~

.. autoclass:: mdpax.utils.checkpointing.CheckpointMixin
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members: _restore_state_from_checkpoint

Spaces
~~~~~~

.. autofunction:: mdpax.utils.spaces.create_range_space 

