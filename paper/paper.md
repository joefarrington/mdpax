---
title: "MDPax: GPU-accelerated MDP solvers in Python with JAX "
tags:
  - Python
  - optimization
  - dynamic programming
  - reinforcement learning
authors:
  - name: Joseph Farrington
    orcid: 0000-0003-4156-3419
    affiliation: 1
    corresponding: True
  - name: Wai Keong Wong
    affiliation: "1, 2, 3"
  - name: Kezhi Li
    affiliation: 1
  - name: Martin Utley
    affiliation: 4
affiliations:
  - name: Institute of Health Informatics, University College London, United Kingdom
    index: 1
  - name: NIHR University College London Hospitals Biomedical Research Centre, United Kingdom
    index: 2
  - name: Cambridge University Hospitals NHS Foundation Trust, United Kingdom
    index: 3
  - name: Clinical Operational Research Unit, University College London, United Kingdom
    index: 4

date: 1 August 2025
bibliography: paper.bib
---

# Summary

MDPax is a Python library for solving large-scale Markov decision processes (MDPs), leveraging JAX’s [@bradbury_jax_2022] support for vectorization, parallelization, and just-in-time (JIT) compilation on graphics processing units (GPUs). It includes GPU-accelerated implementations of standard algorithms including value iteration and policy iteration [@sutton_reinforcement_2018].

MDPs describe sequential decision-making problems in which, at each timestep, an agent observes the current state of its environment, selects an action, transitions to a new state by taking the selected action, and receives a reward. The goal is to find a policy (a mapping from observed states to actions) that maximizes the expected long-term reward, accounting for both the immediate and future consequences of actions. MDPs have been used to model a wide range of problems, including medical treatment planning [@schaefer_modeling_2004], traffic light control [@haijema_dynamic_2017], financial portfolio management [@bauerle_markov_2011], and conservation policy [@nicol_conservation_2010].

Exact solution methods based on dynamic programming scale poorly due to the curse of dimensionality [@bellman_dynamic_1957]: the number of states can grow exponentially with problem parameters, quickly making computation very challenging. Deep reinforcement learning has proven to be very effective at finding approximate solutions for large problems [@arulkumaran_deep_2017], but exact solutions remain valuable both in their own right and for benchmarking approximate methods.

The exact algorithms are well suited to parallel execution, and modern GPUs have thousands of processing cores over which updates can be effectively distributed. By building on JAX, MDPax makes it easy to take advantage of this hardware through a high-level Python API, enabling researchers and practitioners to solve problems with millions of states without needing expertise in GPU programming.

# Statement of need

MDPax was originally developed to support our work on perishable inventory management. Solving these problems exactly requires accounting not just for total inventory levels, but also for the age profile of the stock. As a result, the state space grows exponentially with the product’s maximum useful life, making realistic problem instances extremely large. Although many MDP solvers exist, exact methods are widely considered impractical or infeasible for realistically sized perishable inventory problems [@nahmias_perishable_1982; @hendrix_computing_2019; @de_moor_reward_2022; @abouee-mehrizi_platelet_2025].

Implementations of exact solution methods face two main challenges when applied to large MDPs:

- Memory requirements: The full transition matrix describing the dynamics of the problem grows quadratically with the number of states and linearly with the number of actions, making it infeasible to store and process for large problems.

- Computational complexity: The core operations involve nested loops over states, actions, and successor states, which become prohibitively expensive as the state space grows.

Most existing MDP libraries run exclusively on CPUs. MDPtoolbox [@chades_mdptoolbox_2014] provides exact solution algorithms across Python [@cordwell_pymdptoolbox_2015], MATLAB [@cros_markov_2015], and R [@chades_mdptoolbox_2017], but offers no support for parallelism or GPU acceleration. Two more recent Python libraries, MDPSolver [@andersen_mdpsolver_2025] and madupite [@gargiani_madupite_2025], aim to improve performance on large problems by implementing their solvers in C++ with CPU-based parallelism. In addition, madupite provides a broader set of inexact policy iteration methods to support solving larger problems. For Julia, POMDPs.jl [@egorov_pomdpsjl_2017] provides a flexible interface for specifying MDPs and supports a wide range of CPU-based solvers.

The benefits of GPU-acceleration for exact methods have been demonstrated in the literature [@johannsson_gpu-based_2009; @ortega_cuda_2019; @sargent_quantitative_2025], but remain uncommon in practice. We suggest this is due to the perceived complexity of GPU programming and the limited availability of researcher-friendly software that provides GPU-accelerated solvers. The VFI Toolkit for MATLAB [@kirkby_toolkit_2017] supports GPU-acceleration but requires users to provide the full transition matrix, which becomes infeasible for large problems due to memory constraints.

MDPax addresses the memory challenge by requiring users to provide a deterministic transition function instead of the full transition matrix (similar to the approach used in POMDPs.jl). This function maps a state, action, and random event to the resulting next state and reward, and is used to dynamically compute the next state and reward on demand. MDPax uses JAX to exploit the massive parallel processing capabilities of modern GPUs, significantly reducing the runtime required for solving large MDPs by calculating value updates for batches of states in parallel.

MDPax has been developed to solve large MDPs with millions of states. For small to medium-sized MDPs MDPax may be slower than existing CPU-based packages due to the overheads introduced by the use of JAX and GPUs, including JIT compilation and data transfer between the host and GPU(s). For large problems, these overheads are outweighed by substantial performance gains.

An early version of MDPax was used in our work to solve large instances of three perishable inventory problems that had previously been described as infeasible or impractical to solve exactly [@farrington_going_2025]. In one case, the original study reported that value iteration using a CPU-based MATLAB implementation failed to converge within a week on an MDP with over 16 million states [@hendrix_computing_2019]. Using MDPax the same algorithm converged in under 3.5 hours on a consumer GPU and, with no code changes, in under 30 minutes using four data-centre grade GPUs.

# Features and design

MDPax is structured around two core classes: the Problem and the Solver.

The Problem class represents an MDP and is intended to be subclassed by users. To define a custom problem, users implement methods that specify the sets of states and actions, random events and their probabilities, and a deterministic transition function that maps a (state, action, random event) triple to the next state and corresponding reward. MDPax includes four example Problems: a forest management problem adapted from pymdptoolbox [@cordwell_pymdptoolbox_2015] and three perishable inventory problems from the literature [@hendrix_computing_2019; @de_moor_reward_2022; @abouee-mehrizi_platelet_2025].

The Solver class defines a common framework for implementing dynamic programming methods to solve MDPs. MDPax currently includes implementations of three standard algorithms: value iteration, relative value iteration (to optimize the average reward), and policy iteration. It also provides a variant of value iteration for MDPs with periodic dynamics (e.g. when demand depends on the day of the week), and a semi-asynchronous version in which updated values for each batch of states are made available for subsequent batch updates within the same iteration on the same device.

For large problems, solving an MDP can still be time-consuming. MDPax therefore includes checkpointing functionality using Orbax [@gaffney_orbax_2025], enabling users to save and restore the state of the Solver and resume optimization after an interruption.

# Acknowledgements

JF was funded by UKRI training grant EP/S021612/1, the CDT in AI-enabled Healthcare Systems, and the Clinical and Research Informatics Unit at the NIHR University College London Hospitals Biomedical Research Centre.

# References
