# ADProblems and evaluators

This note describes the current evaluator interface in `AbstractPPL`.

`AbstractPPL` supports evaluators over either named values or flat floating-point vectors.
Downstream packages provide evaluators in one of those two shapes, and `AbstractPPL`
handles backend-specific automatic differentiation setup.

## Supported evaluator shapes

`AbstractPPL` supports two evaluator shapes.

### 1. NamedTuple evaluators

These are callable on named values and are often the most natural fit for model-facing code.

```julia
prepare(problem, prototype::NamedTuple)
prepared(values::NamedTuple)::Real
```

Example:

```julia
values = (x=0.0, y=[1.0, 2.0])
prepared = prepare(problem, values)
value = prepared((x=0.5, y=[1.5, 2.5]))
```

### 2. Vector evaluators

These are callable on floating-point vectors and are useful for optimizers and AD backends
that work with flat parameter vectors.

```julia
prepare(problem, x::AbstractVector{<:AbstractFloat})
prepared(x::AbstractVector{<:AbstractFloat})::Real
dimension(prepared)::Int
```

Example:

```julia
x0 = zeros(3)
prepared = prepare(problem, x0)
value = prepared([0.5, 1.5, 2.5])
d = dimension(prepared)
```

## Public API

The public interface in `src/ADProblems.jl` is:

```julia
DerivativeOrder{K}
capabilities(T::Type)
capabilities(x)
prepare(problem, values::NamedTuple)
prepare(problem, x::AbstractVector{<:AbstractFloat})
prepare(adtype, problem, values_or_x)
value_and_gradient(prepared, values_or_x)
test_autograd(prepared, x::AbstractVector)
dimension(prepared)
```

`AbstractPPL` does not require an abstract supertype for either `problem` or the prepared
object. Packages define their own concrete types and dispatch on them directly.

## Derivative support

Derivative support is described by the `capabilities` trait:

```julia
capabilities(T::Type) = DerivativeOrder{0}()
capabilities(x) = capabilities(typeof(x))
```

The current meanings are:

  - `DerivativeOrder{0}()` — value only
  - `DerivativeOrder{1}()` — gradients are available through `value_and_gradient`
  - `DerivativeOrder{2}()` — reserved by the trait, but not currently exposed through this interface

## AD backends

AD backends extend the three-argument `prepare` interface:

```julia
prepare(adtype, problem, prototype::NamedTuple)
prepare(adtype, problem, x::AbstractVector{<:AbstractFloat})
```

The prepared result is backend-specific, but the common contract is:

  - it is callable on the shape it was prepared for,
  - `capabilities(prepared)` reports derivative support,
  - `value_and_gradient(prepared, values_or_x)` returns `(value, gradient)`.

Current backend support is:

  - `ForwardDiff`, `FiniteDifferences`, and `Mooncake`: named-tuple and vector paths
  - `Enzyme` and the generic `DifferentiationInterface` extension: vector path only

This keeps downstream packages focused on defining evaluators, while `AbstractPPL` handles
backend-specific AD setup.

## Flattening for vector-based backends

Some backends need a flat vector even when the user-facing evaluator works on named values.
For those cases, `AbstractPPL` provides:

```julia
flatten_to!!(buf, x)
unflatten_to!!(x, buf)
```

These utilities support a limited structural subset:

  - `Real`
  - `Complex`
  - `AbstractArray{<:Union{Real,Complex}}`
  - tuples of supported values
  - named tuples of supported values

That is the subset used by the AD extensions when they need to move between named values and
flat vectors.

## Testing gradients

`AbstractPPL` provides:

```julia
test_autograd(
    prepared, x::AbstractVector; atol=1e-5, rtol=1e-5, finite_difference_kwargs...
)
```

This currently validates vector-based gradient implementations against a finite-difference
reference.
