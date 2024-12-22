Notes:

* Currently integer overflow error when getting policy for Forest example using VIR
# Should make sure in docs it is clear that for action space and event space, if multi-dim actions of events than space should be exhaustive, one entry for each valid combination. Can point to examples, e.g. Hendrix Two Product. 

# FiniteHorizon
# Our version of the Guass-Seidel thing is the semi-async.



# Go through SemiAsync and think about it. Might benefit from being a bit simpler. 

# Set level of precision in logging based on epsilon?

# Weird hashing tracer warning when using Relative value iteration that we don't get with normal VI despite being very similar.

# Check periodic convergence check logic and think about whether we should do with checkpoints etc again/. 

# Could think about adding support for early stopping in terms of policy changes: so, like, extra overhead because you have to get the policy out, but maybe it saves a lot of iterations. Could say, oh, if no policy changes for 5/10 iterations etc then stop. 

# Hendrix
* Think about adding ability to do initial estimate of value function using one step ahead revenue
* There is small difference to comparison, probably because we start at a different point (at zero rather than with expected revenue) and many states are similar
* Can we use slices? Shouldn't need to use dynamic slice - only need that when indices are args to function

Active TODOs:
- Think about output from solve() as dict with policy, iterations, converged and maybe an info dict? USE OUR NEW STATE!
- Checkpoint at convergence if checkpointing enabled. Then can restore the whole object. 
- Verbose arg to solver control logging
- Why JIT (or something else in setup) so much slower for Mirjalili and only taking smaller batch sizes
- Update tests to do checkpointing, restoration etc
- Nice example notebook and readme


