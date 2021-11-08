using AbstractMCMC
using DensityInterface


"""
    AbstractProbabilisticProgram

Common base type for models expressed as probabilistic programs.
"""
abstract type AbstractProbabilisticProgram <: AbstractMCMC.AbstractModel end

DensityInterface.hasdensity(::AbstractProbabilisticProgram) = true


"""
    logdensityof(model, trace)

Evaluate the (possibly unnormalized) density of the model specified by the probabilistic program
in `model`, at specific values for the random variables given through `trace`.

`trace` can be of any supported internal trace type, or a fixed probability expression.

`logdensityof` should interact with conditioning and deconditioning in the way required by
probability theory.
"""
DensityInterface.logdensityof


"""
    decondition(conditioned_model)

Remove the conditioning (i.e., observation data) from `conditioned_model`, turning it into a
generative model over prior and observed variables.

The invariant 

```
m == condition(decondition(m), obs)
```

should hold for models `m` with conditioned variables `obs`.
"""
function decondition end


"""
    condition(model, observations)

Condition the generative model `model` on some observed data, creating a new model of the (possibly
unnormalized) posterior distribution over them.

`observations` can be of any supported internal trace type, or a fixed probability expression.

The invariant 

```
m = decondition(condition(m, obs))
```

should hold for generative models `m` and arbitrary `obs`.
"""
function condition end
