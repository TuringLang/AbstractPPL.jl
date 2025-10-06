# `AbstractProbabilisticProgram` interface

There are at least two somewhat incompatible conventions used for the term "model".  None of this is
particularly exact, but:

  - In Turing.jl, if you write down a `@model` function and call it on arguments, you get a model
    object paired with (a possibly empty set of) observations. This can be treated as instantiated
    "conditioned" object with fixed values for parameters and observations.
  - In Soss.jl, "model" is used for a symbolic "generative" object from which concrete functions, such as
    densities and sampling functions, can be derived, _and_ which you can later condition on (and in
    turn get a conditional density etc.).

Relevant discussions:
[1](https://julialang.zulipchat.com/#narrow/stream/234072-probprog/topic/Naming.20the.20.22likelihood.22.20thingy),
[2](https://github.com/TuringLang/AbstractPPL.jl/discussions/10).

## TL/DR:

There are three interrelating aspects that this interface intends to standardize:

  - Density calculation
  - Sampling
  - "Conversions" between different conditionings of models

Therefore, the interface consists of an `AbstractProbabilisticProgram` supertype, together with
functions

  - `condition(::Model, ::Trace) -> ConditionedModel`
  - `decondition(::ConditionedModel) -> GenerativeModel`
  - `sample(::Model, ::Sampler = Exact(), [Int])` (from `AbstractMCMC.sample`)
  - `logdensityof(::Model, ::Trace)` and `densityof(::Model, ::Trace)` (from
    [DensityInterface.jl](https://github.com/JuliaMath/DensityInterface.jl))

## Traces & probability expressions

First, an infrastructural requirement which we will need below to write things out.

The kinds of models we consider are, at least in a theoretical sense, distributions over *traces* –
types which carry collections of values together with their names.  Existing realizations of these
are `VarInfo` in Turing.jl, choice maps in Gen.jl, and the usage of named tuples in Soss.jl.

Traces solve the problem of having to name random variables in function calls, and in samples from
models.  In essence, every concrete trace type will just be a fancy kind of dictionary from variable
names (ideally, `VarName`s) to values.

Since we have to use this kind of mapping a lot in the specification of the interface, let's for now
just choose some arbitrary macro-like syntax like the following:

```julia
@T(Y[1] = …, Z = …)
```

Some more ideas for this kind of object can be found at the end.

## "Conversions"

The purpose of this part is to provide common names for how we want a model instance to be
understood.  As we have seen, in some modelling languages, model instances are primarily generative,
with some parameters fixed, while other instance types pair model instances conditioned on
observations.  What I call "conversions" here is just an interface to transform between these two
views and unify the involved objects under one language.

Let's start from a generative model with parameter `μ`:

```julia
# (hypothetical) generative spec a la Soss
@generative_model function foo_gen(μ)
    X ~ Normal(0, μ)
    Y[1] ~ Normal(X)
    return Y[2] ~ Normal(X + 1)
end
```

Applying the "constructor" `foo_gen` now means to fix the parameter, and should return a concrete
object of the generative type:

```julia
g = foo_gen(; μ=…)::SomeGenerativeModel
```

With this kind of object, we should be able to sample and calculate joint log-densities from, i.e.,
over the combined trace space of `X`, `Y[1]`, and `Y[2]` – either directly, or by deriving the
respective functions (e.g., by converting form a symbolic representation).

For model types that contain enough structural information, it should then be possible to condition
on observed values and obtain a conditioned model:

```julia
condition(g, @T(Y = …))::SomeConditionedModel
```

For this operation, there will probably exist syntactic sugar in the form of

```julia
g | @T(Y = …)
```

Now, if we start from a Turing.jl-like model instead, with the "observation part" already specified,
we have a situation like this, with the observations `Y` fixed in the instantiation:

```julia
# conditioned spec a la DPPL
@model function foo(Y, μ)
    X ~ Normal(0, μ)
    Y[1] ~ Normal(X)
    return Y[2] ~ Normal(X + 1)
end

m = foo(; Y=…, μ=…)::SomeConditionedModel
```

From this we can, if supported, go back to the generative form via `decondition`, and back via
`condition`:

```julia
decondition(m) == g::SomeGenerativeModel
m == condition(g, @T(Y = …))
```

(with equality in distribution).

In the case of Turing.jl, the object `m` would at the same time contain the information about the
generative and posterior distribution `condition` and `decondition` can simply return different
kinds of "tagged" model types which put the model specification into a certain context.

Soss.jl pretty much already works like the examples above, with one model object being either a
`JointModel` or a `ConditionedModel`, and the `|` syntax just being sugar for the latter.

A hypothetical `DensityModel`, or something like the types from LogDensityProblems.jl, would be a
case for a model type that does not support the structural operations `condition` and
`decondition`.

The invariances between these operations should follow normal rules of probability theory.  Not all
methods or directions need to be supported for every modelling language; in this case, a
`MethodError` or some other runtime error should be raised.

There is no strict requirement for generative models and conditioned models to have different types
or be tagged with variable names etc.  This is a choice to be made by the concrete implementation.

Decomposing models into prior and observation distributions is not yet specified; the former is
rather easy, since it is only a marginal of the generative distribution, while the latter requires
more structural information.  Perhaps both can be generalized under the `query` function I discuss
at the end.

## Sampling

Sampling in this case refers to producing values from the distribution specified in a model
instance, either following the distribution exactly, or approximating it through a Monte Carlo
algorithm.

All sampleable model instances are assumed to implement the `AbstractMCMC` interface – i.e., at
least [`step`](https://github.com/TuringLang/AbstractMCMC.jl#sampling-step), and accordingly
`sample`, `steps`, `Samples`.  The most important aspect is `sample`, though, which plays the role
of `rand` for distributions.

The results of `sample` generalize `rand` – while `rand(d, N)` is assumed to give you iid samples,
`sample(m, sampler, N)` returns a sample from a sequence (known as chain in the case of MCMC) of
length `N` approximating `m`'s distribution by a specific sampling algorithm (which of course
subsumes the case that `m` can be sampled from exactly, in which case the "chain" actually is iid).

Depending on which kind of sampling is supported, several methods may be supported.  In the case of
a (posterior) conditioned model with no known sampling procedure, we just have what is given through
`AbstractMCMC`:

```julia
sample([rng], m, N, sampler; [args]) # chain of length N using `sampler`
```

In the case of a generative model, or a posterior model with exact solution, we can have some more
methods without the need to specify a sampler:

```julia
sample([rng], m; [args])    # one random sample
sample([rng], m, N; [args]) # N iid samples; equivalent to `rand` in certain cases
```

It should be possible to implement this by a special sampler, say, `Exact` (name still to be
discussed), that can then also be reused for generative sampling:

```
step(g, spl = Exact(), state = nothing) # IID sample from exact distribution with trivial state
sample(g, Exact(), [N]) 
```

with dispatch failing for models types for which exact sampling is not possible (or not
implemented).

This could even be useful for Monte Carlo methods not being based on Markov Chains, e.g.,
particle-based sampling using a return type with weights, or rejection sampling.

Not all variants need to be supported – for example, a posterior model might not support
`sample(m)` when exact sampling is not possible, only `sample(m, N, alg)` for Markov chains.

`rand` is then just a special case when "trivial" exact sampling works for a model, e.g. a joint
model.

## Density Evaluation

Since the different "versions" of how a model is to be understood as generative or conditioned are
to be expressed in the type or dispatch they support, there should be no need for separate functions
`logjoint`, `loglikelihood`, etc., which force these semantic distinctions on the implementor; we
therefore adapt the interface of
[DensityInterface.jl](https://github.com/JuliaMath/DensityInterface.jl).  Its main function
`logdensityof` should suffice for variants, with the distinction being made by the capabilities of
the concrete model instance.

DensityInterface.jl also requires the trait function `DensityKind`, which is set to `HasDensity()`
for the `AbstractProbabilisticProgram` type.  Additional functions

```
DensityInterface.densityof(d, x) = exp(logdensityof(d, x))
DensityInterface.logdensityof(d) = Base.Fix1(logdensityof, d)
DensityInterface.densityof(d) = Base.Fix1(densityof, d)
```

are provided automatically (repeated here for clarity).

Note that `logdensityof` strictly generalizes `logpdf`, since the posterior density will of course
in general be unnormalized and hence not a probability density.

The evaluation will usually work with the internal, concrete trace type, like `VarInfo` in
Turing.jl:

```julia
logdensityof(m, vi)
```

But the user will more likely work on the interface using probability expressions:

```julia
logdensityof(m, @T(X = …))
```

(Note that this would replace the current `prob` string macro in Turing.jl.)

Densities need (and usually, will) not be normalized.

### Implementation notes

It should be able to make this fall back on the internal method with the right definition and
implementation of `maketrace`:

```julia
logdensityof(m, t::ProbabilityExpression) = logdensityof(m, maketrace(m, t))
```

There is one open question – should normalized and unnormalized densities be able to be
distinguished?  This could be done by dispatch as well, e.g., if the caller wants to make sure
normalization:

```
logdensityof(g, @T(X = …, Y = …, Z = …); normalized=Val{true})
```

Although there is proably a better way through traits; maybe like for arrays, with
`NormalizationStyle(g, t) = IsNormalized()`?

## More on probability expressions

Note that this needs to be a macro, if written this way, since the keys may themselves be more
complex than just symbols (e.g., indexed variables.)  (Don't hang yourselves up on that `@T` name
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
  - "Graph queries": `@T(X | Parents(X))`, `@T(Y | Not(X))` (a nice way to express Gibbs conditionals!)
  - Predicate style for "measure queries": `@T(X < Y + Z)`

The latter applications are the reason I originally liked the idea of the macro being called `@P`
(or even `@𝓅` or `@ℙ`), since then it would look like a "Bayesian probability expression": `@P(X < Y + Z)`.  But this would not be so meaningful in the case of representing a trace instance.

Perhaps both `@T` and `@P` can coexist, and both produce different kinds of `ProbabilityExpression`
objects?

NB: the exact details of this kind of "schema application", and what results from it, will need to
be specified in the interface of `AbstractModelTrace`, aka "the new `VarInfo`".