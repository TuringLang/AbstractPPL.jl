# AbstractPPL.jl

[![CI](https://github.com/TuringLang/AbstractPPL.jl/workflows/CI/badge.svg?branch=master)](https://github.com/TuringLang/AbstractPPL.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![IntegrationTest](https://github.com/TuringLang/AbstractPPL.jl/workflows/IntegrationTest/badge.svg?branch=master)](https://github.com/TuringLang/AbstractPPL.jl/actions?query=workflow%3AIntegrationTest+branch%3Amaster)
[![Coverage Status](https://coveralls.io/repos/github/TuringLang/AbstractPPL.jl/badge.svg?branch=master)](https://coveralls.io/github/TuringLang/AbstractPPL.jl?branch=master)
[![Codecov](https://codecov.io/gh/TuringLang/AbstractPPL.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/TuringLang/AbstractPPL.jl)

A new light-weight package to factor out interfaces and associated APIs for probabilistic
programming languages (especially their modeling languages).  The overall goals are creating an
abstract type and minimal set of functions that will be supported all model and trace types.  Some
other commonly used code, such as variable names, can also go here.


## `AbstractProbabilisticProgram` interface

There are at least two incompatible conventions used for the term “model”: in Turing.jl, it is an
instantiated “conditional distribution” object with fixed values for parameters and observations,
while in Soss.jl, it is the raw symbolic structure from which distributions can be derived.

Relevant discussions:
[1](https://julialang.zulipchat.com/#narrow/stream/234072-probprog/topic/Naming.20the.20.22likelihood.22.20thingy), [2](https://github.com/TuringLang/AbstractPPL.jl/discussions/10).


### Traces & probability expressions

Models are always, at least in a theoretical sense, distributions over *traces* – types which carry
collections of values together with their names.  Existing realizations of these are `VarInfo` in
Turing.jl, choice maps in Gen.jl, and the usage of named tuples in Soss.jl.

Traces solve the problem of having to name random variables in function calls, and in samples from
models.  In essence, every concrete trace type will just be a fancy kind of dictionary from variable
names (ideally, `VarName`s) to values.

```julia
t = @T(Y[1] = ..., Z = ...)
```

Note that this needs to be a macro, if written this way, since the keys may themselves be more
complex than just symbols (e.g., indexed variables.)  (Don’t hang yourselves up on that `@T` name
though, this is just a working draft.)

The idea here is to standardize the construction (and manipulation) of *abstract probability
expressions*, plus the interface for turning them into concrete traces for a specific model – like
[`@formula`](https://juliastats.org/StatsModels.jl/stable/formula/#Modeling-tabular-data) and
[`apply_schema`](https://juliastats.org/StatsModels.jl/stable/internals/#Semantics-time-(apply_schema))
from StatsModels.jl are doing.

Maybe the following would suffice to do that:

```julia
maketrace(m, t)::tracetype(m, t)
```

where `maketrace` produces a concrete trace corresponding to `t` for the model `m`, and `tracetype`
is the corresponding `eltype`–like function giving you the concrete trace type for a certain model
and probability expression combination.

Possible extensions of this idea:

- Pearl-style do-notation: `@T(Y = y | do(X = x))`
- Allowing free variables, to specify model transformations: `query(m, @T(X | Y))`
- “Graph queries”: `@T(X | Parents(X))`, `@T(Y | Not(X))` (a nice way to express Gibbs conditionals!)
- Predicate style for “measure queries”: `@T(X < Y + Z)`

The latter applications are the reason I originally liked the idea of the macro being called `@P`
(or even `@𝓅` or `@ℙ`), since then it would look like a “Bayesian probability expression”: `@P(X <
Y + Z)`.  But this would not be so meaningful in the case of representing a trace instance.

Perhaps both `@T` and `@P` can coexist, and both produce different kinds of `ProbabilityExpression`
objects?

NB: the exact details of this kind of “schema application”, and what results from it, will need to
be specified in the interface of `AbstractModelTrace`, aka “the new `VarInfo`”.


### “Conversions”

The purpose of this part is to provide common names for how we want a model instance to be
understood.  In some modelling languages, model instances are primarily generative or “joint”, with
some parameters fixed (e.g. in Soss.jl), while other instance types pair model instances conditioned
on observations (e.g. Turing.jl’s models).

Let’s start from a generative model:

```julia
# (hypothetical) generative spec a la Soss
@generativemodel function foo_gen(μ)
    X ~ Normal(0, μ)
    Y[1] ~ Normal(X)
    Y[2] ~ Normal(X + 1)
end
```

Applying the “constructor” `foo_gen` now means to fix the parameters, and should return a concrete
object of the generative type (a `JointDistribution` in Soss.jl):

```julia
g = foo_gen(μ=…)::GenerativeModel
```

With this kind of object, we should be able to sample and calculate joint log-densities from, i.e.,
over the combined trace space of `X`, `Y[1]`, and `Y[2]`.

For model types that contain enough structural information, it should then be possible to condition
on observed values and obtain a conditioned model:

```julia
condition(g, @T(Y = ...))::ConditionedModel
```

For this operation, there will probably exist syntactic sugar in the form of

```julia
g | @T(Y = ...)
```

Now, if we start from a Turing.jl-like model instead, with the “observation part” already specified,
we have a situation like this, with the observation fixed in the instantiation:

```julia
# conditioned spec a la DPPL
@model function foo(Y, μ)
    X ~ Normal(0, μ)
    Y[1] ~ Normal(X)
    Y[2] ~ Normal(X + 1)
end

m = foo(Y=…, μ=…)::ConditionedModel
```

From this we can, if supported, go back to the generative form via `decondition`, and back via `condition`:

```julia
decondition(m) == g::GenerativeModel
m == condition(g, @T(Y = ...))
```


In the case of Turing.jl, the object `m` would at the same time contain the information about the
generative and posterior distribution `condition` and `decondition` can simply return different
kinds of “tagged” model types which put the model specification into a certain context.

Soss.jl pretty much already works like the examples above, with one model object being either a
`JointModel` or a `ConditionedModel`, and the `|` syntax just being sugar for the latter.

A hypothetical `DensityModel`, or something like the types from LogDensityProblems.jl, would be a
case for a model type that does not support the structural operations `condition` and
`decondition`. 


### Sampling

For sampling, model instances are assumed to implement the `AbstractMCMC` interface – i.e., at least
[`step`](https://github.com/TuringLang/AbstractMCMC.jl#sampling-step), and accordingly `sample`,
`steps`, `Samples`.  The most important aspect is `sample`, though, which plays the role of `rand`
for distributions.

The results of `sample` generalize `rand` – while `rand(d, N)` is assumed to give you iid samples,
`sample(m, sampler, N)` returns a sample from a (Markov) chain of length `N` approximating `m`’s
distribution by a specific sampling algorithm (which of course subsumes the case that `m` can be
sampled from exactly, in which case the “chain” actually is iid).

Depending on which kind of sampling is supported, several methods may be supported.  In the case of
a (posterior) `ConditionedModel` with no known exact sampling possible, we just have what is given
through `AbstractMCMC`:

```julia
sample([rng], m, N, sampler; [args…]) # chain of length N using `sampler`
```

In the case of a generative model, or a posterior model with exact solution, we can have some more
methods without the need to specify a sampler:

```julia
sample([rng], m; [args…])    # one random sample
sample([rng], m, N; [args…]) # N iid samples; equivalent to `rand` in certain cases
```

It should be possible to implement this by a special sampler `Exact` (name still to be discussed),
that can then also be reused for generative sampling:

```
step(g, spl = Exact(), state = nothing) # IID sample from exact distribution with trivial state
sample(g, Exact(), [N]) 
```

with dispatch failing for models types for which exact sampling is not possible (or implemented).

This could even be useful for Monte Carlo methods not being based on Markov Chains, e.g.,
particle-based sampling using a return type with weights, or rejection sampling.

Not all variants need to be supported – for example, a posterior model might not support
`sample(m)` when exact sampling is not possible, only `sample(m, N, alg)` for Markov chains.

`rand` is then just a special case when “trivial” exact sampling works for a model, e.g. a joint
model.


### Density Calculation

Since the different “contexts” of how a model is to be understood are to be expressed in the type,
there should be no need for separate functions `logjoint`, `loglikelihood`, etc., but one
`logdensity` suffice for all.  Note that this generalizes `logpdf`, too, since the posterior density
will of course in general be unnormalized.

The evaluation will usually work with the internal, concrete trace type, like `VarInfo` in Turing.jl:

```julia
logdensity(m, vi)
```

But the user will more likely work on the interface using probability expressions:

```julia
logdensity(m, @T(X = ...))
```

(Note that this could replace the current `prob` string macro in Turing.jl.)

It should be able to make this fall back on the internal method with the right definition and
implementation of `maketrace`:

```julia
logdensity(m, t::ProbabilityExpression) = logdensity(m, maketrace(m, t))
```

There is one open question – should normalized and unnormalized densities be able to be
distinguished?  This could be done by dispatch as well, e.g., if the caller wants to make sure normalization:

```
logdensity(g, @T(X = ..., Y = ..., Z = ...); normalized=Val{true})
```

Although there is proably a better way through traits; maybe like for arrays, with
`NormalizationStyle(g, t) = IsNormalized()`?


## TL/DR:

- Probability expressions: `@T` and `maketrace`
- `condition(::Model, ::Trace) -> ConditionedModel`
- `decondition(::ConditionedModel) -> GenerativeModel`
- `sample(::Model, ::Sampler = Exact(), [Int])`
- `logdensity(::Model, ::Trace)`

Decomposing models into prior and observation distributions is not yet specified; the former is
rather easy, since it is only a marginal of the generative distribution, while the latter requires
more structural information.  Perhaps both can be generalized under the `query` function I have
hinted to above.






