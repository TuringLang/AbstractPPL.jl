# AbstractPPL.jl

[![CI](https://github.com/TuringLang/AbstractPPL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TuringLang/AbstractPPL.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![IntegrationTest](https://github.com/TuringLang/AbstractPPL.jl/actions/workflows/IntegrationTest.yml/badge.svg?branch=main)](https://github.com/TuringLang/AbstractPPL.jl/actions/workflows/IntegrationTest.yml?query=branch%3Amain)
[![Coverage Status](https://coveralls.io/repos/github/TuringLang/AbstractPPL.jl/badge.svg?branch=main)](https://coveralls.io/github/TuringLang/AbstractPPL.jl?branch=main)
[![Codecov](https://codecov.io/gh/TuringLang/AbstractPPL.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/TuringLang/AbstractPPL.jl)

A light-weight package to factor out interfaces and associated APIs for modelling languages for
probabilistic programming.  High level goals are:

  - Definition of an interface of few abstract types and a small set of functions that should be
    supported by all [probabilistic programs](./src/abstractprobprog.jl) and [trace
    types](./src/abstractmodeltrace.jl).
  - Provision of some commonly used functionality and data structures, e.g., for managing [variable names](./src/varname.jl) and
    traces.

This should facilitate reuse of functions in modelling languages, to allow end users to handle
models in a consistent way, and to simplify interaction between different languages and sampler
implementations, from very rich, dynamic languages like Turing.jl to highly constrained or
simplified models such as GPs, GLMs, or plain log-density problems.

A more short term goal is to start a process of cleanly refactoring and justifying parts of
DynamicPPL.jl’s design, and hopefully to get on closer terms with Soss.jl.

## `AbstractProbabilisticProgram` interface (still somewhat drafty)

There are at least two somewhat incompatible conventions used for the term “model”.  None of this is
particularly exact, but:

  - In Turing.jl, if you write down a `@model` function and call it on arguments, you get a model
    object paired with (a possibly empty set of) observations. This can be treated as instantiated
    “conditioned” object with fixed values for parameters and observations.
  - In Soss.jl, “model” is used for a symbolic “generative” object from which concrete functions, such as
    densities and sampling functions, can be derived, _and_ which you can later condition on (and in
    turn get a conditional density etc.).

Relevant discussions:
[1](https://julialang.zulipchat.com/#narrow/stream/234072-probprog/topic/Naming.20the.20.22likelihood.22.20thingy),
[2](https://github.com/TuringLang/AbstractPPL.jl/discussions/10).

### TL/DR:

There are three interrelating aspects that this interface intends to standardize:

  - Density calculation
  - Sampling
  - “Conversions” between different conditionings of models

Therefore, the interface consists of an `AbstractProbabilisticProgram` supertype, together with
functions

  - `condition(::Model, ::Trace) -> ConditionedModel`
  - `decondition(::ConditionedModel) -> GenerativeModel`
  - `sample(::Model, ::Sampler = Exact(), [Int])` (from `AbstractMCMC.sample`)
  - `logdensityof(::Model, ::Trace)` and `densityof(::Model, ::Trace)` (from
    [DensityInterface.jl](https://github.com/JuliaMath/DensityInterface.jl))

### Traces & probability expressions

First, an infrastructural requirement which we will need below to write things out.

The kinds of models we consider are, at least in a theoretical sense, distributions over *traces* –
types which carry collections of values together with their names.  Existing realizations of these
are `VarInfo` in Turing.jl, choice maps in Gen.jl, and the usage of named tuples in Soss.jl.

Traces solve the problem of having to name random variables in function calls, and in samples from
models.  In essence, every concrete trace type will just be a fancy kind of dictionary from variable
names (ideally, `VarName`s) to values.

Since we have to use this kind of mapping a lot in the specification of the interface, let’s for now
just choose some arbitrary macro-like syntax like the following:

```julia
@T(Y[1] = …, Z = …)
```

Some more ideas for this kind of object can be found at the end.

### “Conversions”

The purpose of this part is to provide common names for how we want a model instance to be
understood.  As we have seen, in some modelling languages, model instances are primarily generative,
with some parameters fixed, while other instance types pair model instances conditioned on
observations.  What I call “conversions” here is just an interface to transform between these two
views and unify the involved objects under one language.

Let’s start from a generative model with parameter `μ`:

```julia
# (hypothetical) generative spec a la Soss
@generative_model function foo_gen(μ)
    X ~ Normal(0, μ)
    Y[1] ~ Normal(X)
    return Y[2] ~ Normal(X + 1)
end
```

Applying the “constructor” `foo_gen` now means to fix the parameter, and should return a concrete
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

Now, if we start from a Turing.jl-like model instead, with the “observation part” already specified,
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
kinds of “tagged” model types which put the model specification into a certain context.

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

### Sampling

Sampling in this case refers to producing values from the distribution specified in a model
instance, either following the distribution exactly, or approximating it through a Monte Carlo
algorithm.

All sampleable model instances are assumed to implement the `AbstractMCMC` interface – i.e., at
least [`step`](https://github.com/TuringLang/AbstractMCMC.jl#sampling-step), and accordingly
`sample`, `steps`, `Samples`.  The most important aspect is `sample`, though, which plays the role
of `rand` for distributions.

The results of `sample` generalize `rand` – while `rand(d, N)` is assumed to give you iid samples,
`sample(m, sampler, N)` returns a sample from a sequence (known as chain in the case of MCMC) of
length `N` approximating `m`’s distribution by a specific sampling algorithm (which of course
subsumes the case that `m` can be sampled from exactly, in which case the “chain” actually is iid).

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

`rand` is then just a special case when “trivial” exact sampling works for a model, e.g. a joint
model.

### Density Evaluation

Since the different “versions” of how a model is to be understood as generative or conditioned are
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

#### Implementation notes

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
(or even `@𝓅` or `@ℙ`), since then it would look like a “Bayesian probability expression”: `@P(X < Y + Z)`.  But this would not be so meaningful in the case of representing a trace instance.

Perhaps both `@T` and `@P` can coexist, and both produce different kinds of `ProbabilityExpression`
objects?

NB: the exact details of this kind of “schema application”, and what results from it, will need to
be specified in the interface of `AbstractModelTrace`, aka “the new `VarInfo`”.

# `AbstractModelTrace`/`VarInfo` interface draft

**This part is even draftier than the above – we’ll try out things in DynamicPPL.jl first**

## Background

### Why do we do this?

As I have said before:

> There are many aspects that make VarInfo a very complex data structure.

Currently, there is an insane amount of complexity and implementation details in DynamicPPL.jl’s
`varinfo.jl`, which has been rewritten multiple times with different concerns in mind – most times
to improve concrete needs of Turing.jl, such as type stability, or requirements of specific
samplers.

This unfortunately makes `VarInfo` extremely opaque: it is hard to refactor without breaking
anything (nobody really dares touching it), and a lot of knowledge about Turing.jl/DynamicPPL.jl
internals is needed in order to judge the effects of changes.

### Design choices

Recently, @torfjelde [has shown](https://github.com/TuringLang/DynamicPPL.jl/pull/267/files) that a
much simpler implementation is feasible – basically, just a wrapped `NamedTuple` with a minimal
interface.

The purpose of this proposal is twofold: first, to think about what a sufficient interface for
`AbstractModelTrace`, the abstract supertype of `VarInfo`, should be, to allow multiple specialized
variants and refactor the existing ones (typed/untyped and simple).  Second, to view the problem as
the design of an abstract data type: the specification of construction and modification mechanisms
for a dictionary-like structure.

Related previous discussions:

  - [Discussion about `VarName`](https://github.com/TuringLang/AbstractPPL.jl/discussions/7)
  - [`AbstractVarInfo` representation](https://github.com/TuringLang/AbstractPPL.jl/discussions/5)

Additionally (but closely related), the second part tries to formalize the “subsumption” mechanism
of `VarName`s, and its interaction with using `VarName`s as keys/indices.

Our discussions take place in what is a bit of a fuzzy zone between the part that is really
“abstract”, and meant for the wider purpuse of AbstractPPL.jl – the implementation of probabilistic
programming systems in general – and our concrete needs within DPPL.  I hope to always stay abstract
and reusable; and there are already a couple of candidates for APPL clients other than DPPL, which
will hopefully keep us focused: simulation based calibration, SimplePPL (a BUGS-like frontend), and
ParetoSmoothing.jl.

### What is going to change?

  - For the end user of Turing.jl: nothing.  You usually don’t use `VarInfo`, or the raw evaluator
    interface, anyways.  (Although if the newer data structures are more user-friendly, they might occur
    in more places in the future?)
  - For people having a look into code using `VarInfo`, or starting to hack on Turing.jl/DPPL.jl: a
    huge reduction in cognitive complexity.  `VarInfo` implementations should be readable on their own,
    and the implemented functions layed out somewhere.  Its usages should look like for any other nice,
    normal data structure.
  - For core DPPL.jl implementors: same as the previous, plus: a standard against which to improve and
    test `VarInfo`, and a clearly defined design space for new data structures.
  - For AbstractPPL.jl clients/PPL implementors: an interface to program against (as with the rest of
    APPL), and an existing set of well-specified, flexible trace data types with different
    characteristics.

And in terms of implementation work in DPPL.jl: once the interface is fixed (or even during fixing
it), varinfo.jl will undergo a heavy refactoring – which should make it _simpler_! (No three
different getter functions with slightly different semantics, etc…).

## Property interface

The basic idea is for all `VarInfo`s to behave like ordered dictionaries with `VarName` keys – all
common operations should just work.  There are two things that make them more special, though:

 1. “Fancy indexing”: since `VarName`s are structured themselves, the `VarInfo` should be have a bit
    like a trie, in the sense that all prefixes of stored keys should be retrievable.  Also,
    subsumption of `VarName`s should be respected (see end of this document):
    
    ```julia
    vi[@varname(x.a)] = [1, 2, 3]
    vi[@varname(x.b)] = [4, 5, 6]
    vi[@varname(x.a[2])] == 2
    vi[@varname(x)] == (; a=[1, 2, 3], b=[4, 5, 6])
    ```
    
    Generalizations that go beyond simple cases (those that you can imagine by storing individual
    `setfield!`s in a tree) need not be implemented in the beginning; e.g.,
    
    ```julia
    vi[@varname(x[1])] = 1
    vi[@varname(x[2])] = 2
    keys(vi) == [x[1], x[2]]
    
    vi[@varname(x)] = [1, 2]
    keys(vi) == [x]
    ```

 2. (_This has to be discussed further._)  Information other than the sampled values, such as flags,
    metadata, pointwise likelihoods, etc., can in principle be stored in multiple of these “`VarInfo`
    dicts” with parallel structure.  For efficiency, it is thinkable to devise a design such that
    multiple fields can be stored under the same indexing structure.
    
    ```julia
    vi[@varname(x[1])] == 1
    vi[@varname(x[1])].meta["bla"] == false
    ```
    
    or something in that direction.
    
    (This is logically equivalent to a dictionary with named tuple values.  Maybe we can do what
    [`DictTable`](https://github.com/JuliaData/TypedTables.jl/blob/main/src/DictTable.jl) does?)
    
    The old `order` field, indicating at which position in the evaluator function a variable has
    been added (essentially a counter of insertions) can actually be left out completely, since the
    dictionary is specified to be ordered by insertion.
    
    The important question here is: should the “joint data structure” behave like a dictionary of
    `NamedTuple`s (`eltype(vi) == @NamedTuple{value::T, ℓ::Float64, meta}`), or like a struct of
    dicts with shared keys (`eltype(vi.value) <: T`, `eltype(vi.ℓ) <: Float64`, …)?

The required dictionary functions are about the following:

  - Pure functions:
    
      + `iterate`, yielding pairs of `VarName` and the stored value
      + `IteratorEltype == HasEltype()`, `IteratorSize = HasLength()`
      + `keys`, `values`, `pairs`, `length` consistent with `iterate`
      + `eltype`, `keytype`, `valuetype`
      + `get`, `getindex`, `haskey` for indexing by `VarName`
      + `merge` to join two `VarInfo`s

  - Mutating functions:
    
      + `insert!!`, `set!!`
      + `merge!!` to add and join elements (TODO: think about `merge`)
      + `setindex!!`
      + `empty!!`, `delete!!`, `unset!!` (_Are these really used anywhere? Not having them makes persistent
        implementations much easier!_)

I believe that adopting the interface of
[Dictionaries.jl](https://github.com/andyferris/Dictionaries.jl), not `Base.AbstractDict`, would be
ideal, since their approach make key sharing and certain operations naturally easy (particularly
“broadcast-style”, i.e., transformations on the values, but not the keys).

Other `Base` functions, like `enumerate`, should follow from the above.

`length` might appear weird – but it should definitely be consistent with the iterator.

It would be really cool if `merge` supported the combination of distinct types of implementations,
e.g., a dynamic and a tuple-based part.

To support both mutable and immutable/persistent implementations, let’s require consistent
BangBang.jl style mutators throughout.

## Transformations/Bijectors

Transformations should ideally be handled explicitely and from outside: automatically by the
compiler macro, or at the places required by samplers.

Implementation-wise, they can probably be expressed as folds?

```julia
map(v -> link(v.dist, v.value), vi)
```

## Linearization

There are multiple possible approaches to handle this:

 1. As a special case of conversion: `Vector(vi)`
 2. `copy!(vals_array, vi)`.
 3. As a fold: `mapreduce(v -> vec(v.value), append!, vi, init=Float64[])`

Also here, I think that the best implementation would be through a fold.  Variants (1) or (2) might
additionally be provided as syntactic sugar.

* * *

# `VarName`-based axioms

What follows is mostly an attempt to formalize subsumption.

First, remember that in Turing.jl we can always work with _concretized_ `VarName`s: `begin`/`end`,
`:`, and boolean indexing are all turned into some form of concrete cartesian or array indexing
(assuming [this suggestion](https://github.com/TuringLang/AbstractPPL.jl/issues/35) being
implemented).  This makes all index comparisons static.

Now, `VarName`s have a compositional structure: they can be built by composing a root variable with
more and more lenses (`VarName{v}()` starts off with an `IdentityLens`):

```julia
julia> vn = VarName{:x}() ∘ Setfield.IndexLens((1:10, 1) ∘ Setfield.IndexLens((2,)))
x[1:10,1][2]
```

(_Note that the composition function, `∘`, is really in wrong order; but this is a heritage of
Setfield.jl._)

By “subsumption”, we mean the notion of a `VarName` expressing a more nested path than another one:

```julia
subsumes(@varname(x.a), @varname(x.a[1]))
@varname(x.a) ⊒ @varname(x.a[1]) # \sqsupseteq
@varname(x.a) ⋢ @varname(x.a[1]) # \nsqsubseteq
```

Thus, we have the following axioms for `VarName`s (“variables” are `VarName{n}()`):

 1. `x ⊑ x` for all variables `x`
 2. `x ≍ y` for `x ≠ y` (i.e., distinct variables are incomparable; `x ⋢ y` and `y ⋢ x`) (`≍` is `\asymp`)
 3. `x ∘ ℓ ⊑ x` for all variables `x` and lenses `ℓ`
 4. `x ∘ ℓ₁ ⊑ x ∘ ℓ₂ ⇔ ℓ₁ ⊑ ℓ₂`

For the last axiom to work, we also have to define subsumption of individual, non-composed lenses:

 1. `PropertyLens(a) == PropertyLens(b) ⇔ a == b`, for all symbols `a`, `b`
 2. `FunctionLens(f) == FunctionLens(g) ⇔ f == g` (under extensional equality; I’m only mentioning
    this in case we ever generalize to Bijector-ed variables like `@varname(log(x))`)
 3. `IndexLens(ι₁) ⊑ IndexLens(ι₂)` if the index tuple `ι₂` covers all indices in `ι₁`; for example,
    `_[1, 2:10] ⊑ _[1:10, 1:20]`.  (_This is a bit fuzzy and not all corner cases have been
    considered yet!_)
 4. `IdentityLens() == IdentityLens()`
 5. `ℓ₁ ≍ ℓ₂`, otherwise

Together, this should make `VarName`s under subsumption a reflexive poset.

The fundamental requirement for `VarInfo`s is then:

```
vi[x ∘ ℓ] == get(vi[x], ℓ)
```

So we always want the following to work, automatically:

```julia
vi = insert!!(vi, vn, x)
vi[vn] == x
```

(the trivial case), and

```julia
x = set!!(x, ℓ₁, a)
x = set!!(x, ℓ₂, b)
vi = insert!!(vi, vn, x)
vi[vn ∘ ℓ₁] == a
vi[vn ∘ ℓ₂] == b
```

since `vn` subsumes both `vn ∘ ℓ₁` and `vn ∘ ℓ₂`.

Whether the opposite case is supported may depend on the implementation.  The most complicated part
is “unification”:

```julia
vi = insert!!(vi, vn ∘ ℓ₁, a)
vi = insert!!(vi, vn ∘ ℓ₂, b)
get(vi[vn], ℓ₁) == a
get(vi[vn], ℓ₂) == b
```

where `vn ∘ ℓ₁` and `vn ∘ ℓ₂` need to be recognized as “children” of a common parent `vn`.
