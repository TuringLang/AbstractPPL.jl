using AbstractMCMC
using Random: AbstractRNG, GLOBAL_RNG
using StatsBase


"""
    AbstractProbabilisticProgram

Common base type for models expressed as probabilistic programs.

A model must support sampling via `generate`/`sample`, and density evaluation by `logdensity` (can
be unnormalized).

Since different PPLs have different opinions about whether data should be attached to a model, this
distinction is captures in the `AbstractContext` parameters of these functions: 
- a "generative" model (e.g., in Soss.jl) expresses a joint density `p(X, ...)` over some variables.
It should implement the interface on `JointContext`.  Such a model does not need distinguish between
observed and latent variables and has no data attached, so we can only ever evaluate and sample from
all variables at once.
- a "conditioned" model (e.g., Turing.jl) fixes some variables to observed values, and therefore
decomposes into a prior (the unobserved variables) and a likelihood (the observed variables given
the parameters): `p(Θ) p(X | Θ)`.  Such a model should implement the interface on `PriorContext` and
`LikelihoodContext`.

A model may also implement any other combination of contexts.
"""
abstract type AbstractProbabilisticProgram <: AbstractMCMC.AbstractModel end


"""
    logdensity(model::AbstractProbabilisticProgram, ctx::AbstractContext, values)

Evalute the log density of the `model` in the mode given by `ctx` (mainly joint, prior, or
likelihood).  The type of `values` is left to the implementation; it should be some representation
of the trace.  `logdensity` should always work in the output of `sample`/`generate`.

Note that a density need not be a probability density.  Consider it unnormalized unless you know
better.

See also [`logjoint`](@ref), [`logprior`](@ref), and [`loglikelihood`](@ref).
"""
function logdensity end


"""
    logjoint(model, values)

Return the log joint probability of for the probabilistic `model`, given all `values`.

See also [`logdensity`](@ref), [`logprior`](@ref), and [`loglikelihood`](@ref).
"""
logjoint(m::AbstractProbabilisticProgram, values) = logdensity(m, JointContext(), values)


"""
    logprior(model, values)

Return the log prior probability of parameters `values` for the probabilistic `model`.

See also [`logdensity`](@ref), [`logjoint`](@ref), and [`loglikelihood`](@ref).
"""1

logprior(m::AbstractProbabilisticProgram, values) = logdensity(m, PriorContext(), values)


"""
    loglikelihood(model, values)

Return the log likelihood 

See also [`logdensity`](@ref), [`logprior`](@ref), and [`loglikelihood`](@ref).
"""
function StatsBase.loglikelihood(m::AbstractProbabilisticProgram, values)
    return logdensity(m, LikelihoodContext(), values)
end


"""
    generate([rng], m::AbstractProbabilisticProgram, ctx::AbstractContext[, args])

Draw a sample from `m` in `ctx` and evaluate its density.  The sampled type is left
to the implementation; it should be some representation of the trace.

Additional `args` are, for example, the sampling algorithm (for approximate sampling) and the
number of samples.  
"""
function generate(
    rng::AbstractRNG,
    m::AbstractProbabilisticProgram,
    ctx::AbstractContext,
    args...)
    s = sample(rng, m, ctx, args...)
    ℓ = logdensity(m, ctx, s)
    return ℓ, s
end

function generate(m::AbstractProbabilisticProgram, ctx::AbstractContext, args...)
    return generate(GLOBAL_RNG, m, ctx, args)
end


"""
    sample([rng], m::AbstractProbabilisticProgram, ctx::AbstractContext[, args])

Draw a sample from the model specified in `m` in the mode given by `ctx`.

Additional `args` are, for example, the sampling algorithm (for approximate sampling) and the
number of samples.
"""
function StatsBase.sample(
    rng::AbstractRNG,
    m::AbstractProbabilisticProgram,
    ctx::AbstractContext,
    args...)
    _, s = generate(rng, m, ctx, args...)
    return s
end

function StatsBase.sample(m::AbstractProbabilisticProgram, ctx::AbstractContext, args...)
    return sample(GLOBAL_RNG, m, ctx, args)
end
