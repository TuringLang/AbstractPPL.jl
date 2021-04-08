# AbstractPPL.jl

[![CI](https://github.com/TuringLang/AbstractPPL.jl/workflows/CI/badge.svg?branch=master)](https://github.com/TuringLang/AbstractPPL.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![IntegrationTest](https://github.com/TuringLang/AbstractPPL.jl/workflows/IntegrationTest/badge.svg?branch=master)](https://github.com/TuringLang/AbstractPPL.jl/actions?query=workflow%3AIntegrationTest+branch%3Amaster)
[![Coverage Status](https://coveralls.io/repos/github/TuringLang/AbstractPPL.jl/badge.svg?branch=master)](https://coveralls.io/github/TuringLang/AbstractPPL.jl?branch=master)
[![Codecov](https://codecov.io/gh/TuringLang/AbstractPPL.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/TuringLang/AbstractPPL.jl)

A new light-weight package to factor out interfaces and associated APIs for probabilistic
programming languages (especially their modelleing languages).  The overall goals are creating an
abstract type and minimal set of functions that will be supported all model and trace types.  Some
other commonly used code, such as variable names, can also go here.


## `AbstractProbabilisticProgram` interface

There are at least two incompatible usages of what “model” means: in Turing.jl, it is the
instantiated object (with fixed values for parameters and observations), while in Soss.jl, it is the
raw symbolic structure.

Relevant discussions:
[1](https://julialang.zulipchat.com/#narrow/stream/234072-probprog/topic/Naming.20the.20.22likelihood.22.20thingy), [2](https://github.com/TuringLang/AbstractPPL.jl/discussions/10).

Assume the model

```julia
X ~ Normal(0, μ)
Y[1] ~ Normal(X)
Y[2] ~ Normal(X + 1)
```

where `μ` is a parameter, `X` is a latent variable, and `Y` are the observations.

The interface has three aspects:
- “Conversions” – the designation of how a probabilistic program is to be understood (joint, prior,
  etc.).
- Sampling functions
- Density calculations


### TODO: Traces

Models are always distribution over *traces* – types which carry collections of values together with
their names.  This solves the problem of naming things in function calls:

```julia
logdensity(m, θ) # P[Θ = θ] in the context of model `m`.
```

```julia
merge(θ, obs) # common trace of two sub-traces – undefined unless disjoint?
```


### “Conversions”

The purpose of this part is to provide common names for how we want a model instance to be
understood.  In some modelling languages, model instances are primarily “joint” or generative, with
some parameters fixed (e.g. in Soss.jl), while other instance types pair model instances with fixed
observations (e.g. Turing.jl’s models).

In the following, assume `m` is a joint model with parameters fixed, and `m | obs` constructs a
conditioned model (with the conditioned variable names specified as part of `obs`):

```julia
prior, obsmodel = splitmodel(m | obs)
priormodel(m | obs) == prior
observationmodel(m | obs) == obsmodel # (θ) ↦ (obs) ↦ p(obs | θ)

likelihoodmodel(m | obs) # (θ) ↦ p(obs | θ), unnormalized model!
loglikelihood((m | obs), θ) == logdensity(observationmodel(m | obs)(θ), obs))

posteriormodel(m | obs) # (θ) ↦ Z * p(θ | obs), unnormalized model!
logposterior((m | obs), θ) == logdensity(priormodel(m | obs), θ) + loglikelihood((m | obs), θ)
# TODO: `logunnormalizedposterior`? `logposterior(m, θ, ::Val{true})`?

jointmodel(m | obs) == m
jointmodel(m) == m
```

The “splitting” functions should simply fail on a model that is not conditioned.  `jointmodel`
should succees in case of a generative as well as a conditioned model, unless a posterior model is
somehow directly implemented without structural splittability (rare case? does this even make sense
as a probabilistic program?).

In the case of Turing.jl, the object `m` would at the same time represent the joint and conditioned
“model”, so `jointmodel` and `splitmodel` return different kinds of “tagged” model types which put
the model specification into a certain context.

Soss.jl pretty much already works like the examples above, with one model object being either a
`JointModel` or a `ConditionedModel`, with the `|` syntax just being sugar for the latter.


### Sampling

For sampling, model instances are assumed to implement the `AbstractMCMC` interface – i.e., at least
[`step`](https://github.com/TuringLang/AbstractMCMC.jl#sampling-step), and accordingly `sample`,
`steps`, `Samples`.  The most important aspect is `sample`, though, which plays the role of `rand`
for distributions.

The results of `sample` generalize `rand` – while `rand(d, N)` is assumed to give you iid samples,
`sample(m, N)` returns a sample from a (Markov) chain of length `N` targetting `m`’s distribution
(which of course subsumes the case that `m` can be sampled from exactly, in which case the “chain”
actually is iid).

Depending on which kind of sampling is needed, several methods may be implemented:

```julia
sample([rng], m; [args…])    # one random sample
sample([rng], m, N; [args…]) # N iid samples; equivalent to `rand` in certain cases
sample([rng], m, N, sampler; [args…]) # chain of length N using `sampler`
```

This could even be useful for Monte Carlo methods not being based on Markov Chains, e.g.,
particle-based sampling using a return type with weights.

Not all variants need to be supported – for example, a posterior model might not support
`sample(m)` when exact sampling is not possible, only `sample(m, N, alg)` for Markov chains.

`rand` is then just a special case when “trivial” forward sampling works for a model, e.g. a joint
model (`rand(m, N, Forward())`?)

We should have

```julia
sample(jointmodel(m)) = begin
    θ = sample(priormodel(m)),
    obs = sample(observationmodel(m | θ))
    merge(θ, obs)
end
```

### Density Calculation

```julia
logdensity(prior(m)) + logdensity()
```



## Scratch


# hypothetical generative spec ala Soss
@generativemodel function foo_gen(μ)
    X ~ Normal(0, μ)
    Y[1] ~ Normal(X)
    Y[2] ~ Normal(X + 1)
end

g = foo(rand())::GenerativeModel
g | @P(Y = ...) == m == condition(g, @P(...))

g | @P(X = ..., Y = ...) # -> model + trace ala Gen

# query(foo_gen, @P(Y | do(X)))

@condition(g, Y = ...) =
    condition(g, P(@varname(Y) => ...))
@decondition(m)

# conditioned spec a la DPPL
@model function foo(Y, μ)
    X ~ Normal(0, μ)
    Y[1] ~ Normal(X)
    Y[2] ~ Normal(X + 1)
end

m = foo(rand(2), rand())::ConditionedModel
decondition(m) == g

logdensity(g, vi)
logdensity(g, @P(X = ..., Y = ..., Z = ...);
           normalized=Val{true}) # warn?

logdensity(m, @P(X = ...)) # unnormalized


step(g, spl = Exact(), state = nothing) = ...
sample(g, Exact(), N) # (X, Y), analytic if possible
sample(decondition(m), N) # (X, Y)

sample(m, spl::Exact) = step(m, spl)

sample(m, spl, N)
sample(m, Exact(), N)

Sample(m, spl)
steps(m, spl)

sample(foo | @P(Y[1] = ...))
maketrace(m, t = @trace(Y[1] = ...))::tracetype(m, t)
