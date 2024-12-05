using AbstractMCMC
using DensityInterface
using Random

"""
    AbstractProbabilisticProgram

Common base type for models expressed as probabilistic programs.
"""
abstract type AbstractProbabilisticProgram <: AbstractMCMC.AbstractModel end

DensityInterface.DensityKind(::AbstractProbabilisticProgram) = HasDensity()

"""
    logdensityof(model, trace)

Evaluate the (possibly unnormalized) density of the model specified by the probabilistic program
in `model`, at specific values for the random variables given through `trace`.

`trace` can be of any supported internal trace type, or a fixed probability expression.

`logdensityof` should interact with conditioning and deconditioning in the way required by
probability theory.
"""
DensityInterface.logdensityof(::AbstractProbabilisticProgram, ::AbstractModelTrace)

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

"""
    fix(model, params)

Fix the values of parameters specified in `params` within the probabilistic model `model`. 
This operation is equivalent to treating the fixed parameters as being drawn from a point mass 
distribution centered at the values specified in `params`. Thus these parameters no longer contribute
to the accumulated log density. 

Conceptually, this is similar to Pearl's do-operator in causal inference, where we intervene 
on variables by setting them to specific values, effectively cutting off their dependencies 
on their usual causes in the model.

The invariant

```
m == unfix(fix(m, params))
```

should hold for any model `m` and parameters `params`.
"""
function fix end

"""
    unfix(model)

Remove any fixed parameters from the model `model`, returning a new model without the fixed parameters.

This function reverses the effect of `fix` by removing parameter constraints that were previously set.
It returns a new model where all previously fixed parameters are allowed to vary according to their 
original distributions in the model.

The invariant

```
m == unfix(fix(m, params))
```

should hold for any model `m` and parameters `params`.
"""
function unfix end

"""
    rand([rng=Random.default_rng()], [T=NamedTuple], model::AbstractProbabilisticProgram) -> T

Draw a sample from the joint distribution of the model specified by the probabilistic program.

The sample will be returned as format specified by `T`.
"""
Base.rand(rng::Random.AbstractRNG, ::Type, model::AbstractProbabilisticProgram)
function Base.rand(rng::Random.AbstractRNG, model::AbstractProbabilisticProgram)
    return rand(rng, NamedTuple, model)
end
function Base.rand(::Type{T}, model::AbstractProbabilisticProgram) where {T}
    return rand(Random.default_rng(), T, model)
end
function Base.rand(model::AbstractProbabilisticProgram)
    return rand(Random.default_rng(), NamedTuple, model)
end
