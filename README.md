# MDPax

[![Documentation Status](https://readthedocs.org/projects/mdpax/badge/?version=latest)](https://mdpax.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/joefarrington/mdpax/actions/workflows/tests.yml/badge.svg)](https://github.com/joefarrington/mdpax/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


MDPax is designed for researchers and practitioners who want to solve large Markov Decision Process (MDP) problems but don't want to become experts in graphics processing unit (GPU) programming. By using JAX, we can take advantage of the massive parallel processing power of GPUs while describing new problems using a simple Python interface.

You can run MDPax on your local GPU, or try it for free using [Google Colab](https://colab.research.google.com/), which provides access to GPUs in the cloud with no setup required.

## Key capabilities
- Solve MDPs with millions of states using value iteration
- Automatic support for one or more identical GPUs
- Flexible interface for defining your own MDP problem or solver algorithm
- Asynchronous checkpointing using [Orbax](https://orbax.readthedocs.io/en/latest/)
- Ready-to-use examples including perishable inventory problems from recent literature

## Overview

MDPax is a Python package for solving large-scale MDPs, leveraging JAX's support for vectorization, parallelization, and just-in-time (JIT) compilation on GPUs. 

The package is adapted from the research code developed in [Farrington et al (2023)](https://arxiv.org/abs/2303.10672). We demonstrated that this approach is particularly well-suited for perishable inventory management problems where the state space grows exponentially with the number of products and the maximum useful life of the products. By implementing the problems in JAX and using consumer-grade GPUs (or freely available GPUs on services such as Google Colab) it is possible to compute the exact solution for realistically sized perishable inventory problems where this was recently reported to be infeasible or impractical.

Traditional value iteration implementations face two main challenges with large state spaces:
1. Memory requirements - the full transition matrix grows with the square of the state space size
2. Computational complexity - nested loops over states, actions, and possible next states become prohibitively expensive

MDPax addresses these challenges by:
1. Using a functional approach where users specify a deterministic transition function in terms of state, action, and random event, rather than providing the full transition matrix
2. Leveraging JAX's transformations to optimize computation:
   - `vmap` to vectorize operations across states and actions
   - `pmap` to parallelize across multiple GPU devices where available
   - `jit` to compile operations once and reuse them efficiently across many value iteration steps

While MDPax can run on CPU or GPU hardware, it is specifically designed for large problems (millions of states) on GPU. For small to medium-sized problems, especially when running on CPU, existing packages like [pymdptoolbox](https://github.com/sawcordwell/pymdptoolbox) may be more efficient due to JAX's JIT compilation overhead and GPU memory transfer costs. These overheads become negligible for larger problems where the benefits of parallelization and vectorization dominate.

## Installation

MDPax can be installed from PyPI using pip:

```bash
pip install mdpax
```
If you want to run the [example notebooks](https://github.com/joefarrington/mdpax/tree/main/examples), install the additional dependencies with:
```bash
pip install "mdpax[examples]"
```

For GPU support, ensure your NVIDIA drivers and CUDA toolkit are compatible with JAX. See the [JAX installation guide](https://github.com/google/jax#installation) for details.

See [`pyproject.toml`](https://github.com/joefarrington/mdpax/blob/main/pyproject.toml) for the complete list of dependencies and version requirements, 

### Google Colab

You can try MDPax without any local installation using Google Colab, which provides free GPU access. See our Getting Started notebook for an interactive introduction. 

To verify you're using a GPU in Colab, click Runtime > Change runtime type and ensure "GPU" is selected as the Hardware accelerator. You can confirm GPU availability by running `!nvidia-smi` in a code cell.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joefarrington/mdpax/blob/main/examples/getting_started.ipynb)

## Quick Start

The following example shows how to solve a simple forest management problem (adapted from [pymdptoolbox's example](https://github.com/sawcordwell/pymdptoolbox?tab=readme-ov-file#quick-use)):

```python
from mdpax.problems import Forest
from mdpax.solvers import ValueIteration

# Create forest management problem
problem = Forest()

# Create solver with discount factor gamma = 0.9,
# and convergence tolerance epsilon = 0.01
solver = ValueIteration(problem, gamma=0.9, epsilon=0.01)

# Solve the problem (automatically uses GPU if available)
solution = solver.solve(max_iterations=500)

# Access the optimal policy and value function
print(solution.policy)  # array([[0], [0], [0]]) - "wait" for all states
print(solution.values)  # value for each state under optimal policy
```

This example demonstrates the core workflow:
1. Create a problem instance
2. Initialize a solver
3. Solve to get the optimal policy and value function

## Documentation

Full documentation is available at [https://mdpax.readthedocs.io/](https://mdpax.readthedocs.io/).

### Tutorials
- [Getting Started](https://mdpax.readthedocs.io/en/latest/notebooks/getting_started.html) - Basic usage on several problems [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joefarrington/mdpax/blob/main/examples/getting_started.ipynb)
- [Creating custom MDP problems](https://mdpax.readthedocs.io/en/latest/notebooks/create_custom_problem.html) - Guide to implementing your own problems, using [FrozenLake](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) as an example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joefarrington/mdpax/blob/main/examples/create_custom_problem.ipynb)

### API Reference
- [Core](https://mdpax.readthedocs.io/en/latest/api.html#core) - The base classes for Problems and Solvers
- [Solvers](https://mdpax.readthedocs.io/en/latest/api.html#solvers) - Value iteration and other solution methods
- [Problems](https://mdpax.readthedocs.io/en/latest/api.html#problems) - Example problems

For reproducible examples from the original paper, see the [viso_jax](https://github.com/joefarrington/viso_jax) repository.

## Example Problems

### Basic example: forest management

A simple forest management problem adapted from [pymdptoolbox](https://github.com/sawcordwell/pymdptoolbox). This problem has a small state space by default (3 states, representing the possible age of the forest) and is useful for getting started with the package or debugging new solvers. The manager must decide whether to cut or wait at each time step, considering the trade-off between immediate revenue from cutting versus letting the forest mature.

### Perishable inventory management problems

These problems demonstrate the package's ability to handle large state spaces in inventory management scenarios and were included in [Farrington et al. (2023)](https://arxiv.org/abs/2303.10672) as examples to demonstrate the benefits of implementing value iteration in JAX.

#### De Moor Single Product Perishable [(De Moor et al. 2022)](https://doi.org/10.1016/j.ejor.2021.10.045)
A single-product inventory system with positive lead time and fixed useful life. Orders placed today arrive after a fixed lead time, and the state must track both current stock levels and orders in transit.

#### Hendrix Two Product Perishable (with substitution) [(Hendrix et al. 2019)](https://doi.org/10.1002/cmm4.1027)
A two-product inventory system with product substitution, where both products have fixed useful lives. Customers may be willing to substitute product A for B when B is out of stock.

#### Mirjalili Platelet Perishable [(Mirjalili 2022; ](https://tspace.library.utoronto.ca/bitstream/1807/124976/1/Mirjalili_Mahdi_202211_PhD_thesis.pdf)[Abouee-Mehrizi et al. 2023)](https://doi.org/10.48550/arXiv.2307.09395)
A single-product inventory management problem, modelling platelet inventory management in a hospital blood bank. Features weekday-dependent demand patterns and uncertain useful life of platelets at arrival, which may depend on the order quantity. 

## Development

To set up a development environment using [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
# Clone the repository
git clone https://github.com/joefarrington/mdpax.git
cd mdpax


# Create a virtual environment
uv venv
source .venv/bin/activate

# Install development dependencies
uv pip install -e ".[dev]"  # Using uv

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest tests/
```

The development environment includes:
- black and ruff for code formatting and linting
- pytest for testing
- pre-commit hooks to ensure code quality
- sphinx for documentation building

See [`pyproject.toml`](https://github.com/joefarrington/mdpax/blob/main/pyproject.toml) for the full list of development dependencies.

## Contributing

MDPax is a new library aimed at researchers and practitioners. As we're in the early stages of development, we particularly welcome feedback on the API design and suggestions for how we can make the library more accessible to users with different backgrounds and experience levels. Our goal is to make using GPUs to solve large MDPs as straightforward as possible while maintaining the flexibility needed for research applications.

Contributions are welcome in many forms:

1. **API and Documentation Feedback**:
   - Is the API intuitive for your use case?
   - Are there concepts that need better explanation?
   - Would additional examples help?
   - Open an issue with your suggestions or questions

2. **Bug Reports**: Open an issue describing:
   - What you were trying to do
   - What you expected to happen
   - What actually happened
   - Steps to reproduce the issue

3. **Feature Requests**: 
    Open an issue describing:
   - The use case for the feature
   - Any relevant references (papers, implementations)
   - Possible implementation approaches

4. **Pull Requests**: For code contributions:
   - Open an issue first to discuss the proposed changes
   - Fork the repository
   - Create a new branch for your feature
   - Follow the existing code style (enforced by pre-commit hooks)
   - Add tests for new functionality
   - Update documentation as needed
   - Submit a PR referencing the original issue

5. **New Problem Implementations**: We're particularly interested in helping users implement new MDP problems:
   - Open an issue describing the problem and citing any relevant papers
   - We can help with the implementation approach and best practices
   - This is a great way to contribute while learning the package

All contributions will be reviewed and should pass the automated checks (tests, linting, type checking).

## Citation

If you use this software in your research, please cite

The original paper:
```bibtex
@misc{farrington2023,
      title={Going faster to see further: GPU-accelerated value iteration and simulation for perishable inventory control using JAX}, 
      author={Joseph Farrington and Kezhi Li and Wai Keong Wong and Martin Utley},
      year={2023},
      eprint={2303.10672},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2303.10672}, 
}
```

The software package:
```bibtex
@software{mdpax2024github,
  author = {Joseph Farrington},
  title = {mdpax: GPU-accelerated MDP solvers in Python with JAX},
  year = {2024},
  url = {https://github.com/joefarrington/mdpax},
}
```

## License

MDPax is released under the MIT License. See the [LICENSE](LICENSE) file for details.

The forest management example problem is adapted from [pymdptoolbox](https://github.com/sawcordwell/pymdptoolbox) (BSD 3-Clause License, Copyright (c) 2011-2013 Steven A. W. Cordwell and Copyright (c) 2009 INRA). Our implementation is original, using the `mdpax.core.problems.Problem` class. 

## Related Projects

### [viso_jax](https://github.com/joefarrington/viso_jax)
The original research code used to produce the results in [Farrington et al. (2023)](https://arxiv.org/abs/2303.10672). Contains implementations of the perishable inventory problems and the experimental setup used in the paper. While MDPax is designed to be a general-purpose library, viso_jax focuses specifically on reproducing the paper's results and includes a detailed Colab notebook for this purpose.

### [Quantitative Economics with JAX](https://jax.quantecon.org/intro.html)
Tutorials using JAX to solve problems from quantitative economics, including value function iteration and policy iteration for MDPs. 

### [VFI Toolkit](https://www.vfitoolkit.com/)
A MATLAB toolkit for value function iteration, specifically in the context of macroeconomic modeling. Like MDPax, the toolkit automatically uses NVIDIA GPUs when available. Unlike MDPax, the toolkit requires the full transition matrix to be provided, which can be infeasible for very large problems. 

### [pymdptoolbox](https://github.com/sawcordwell/pymdptoolbox)
A Python library for solving MDPs that implements several classic algorithms including value iteration, policy iteration, and Q-learning. Related packages are available for MATLAB, GNU Octave, Scilab and R [(Chadès et al. 2014)](https://nsojournals.onlinelibrary.wiley.com/doi/full/10.1111/ecog.00888). pymdptoolbox does not support GPU-acceleration and, like the VFI Toolkit, requires the user to provide the full transition matrix for problems.

## Acknowledgments

This library is based on research code developed during Joseph Farrington's PhD at University College London under the supervision of Ken Li, Martin Utley, and Wai Keong Wong.

The PhD was generously supported by:
- UKRI training grant EP/S021612/1, the CDT in AI-enabled Healthcare Systems
- The Clinical and Research Informatics Unit at the NIHR University College London Hospitals Biomedical Research Centre
