# Evaluator preparation and AD

AbstractPPL provides a small interface for preparing callables and asking a
prepared evaluator for values and derivatives. `prepare` binds a callable to a
sample input that establishes the expected input shape and type;
`value_and_gradient!!` and `value_and_jacobian!!` then return the value and
derivative together.

The `!!` suffix signals that the returned gradient or Jacobian **may alias
internal cache buffers** of the prepared evaluator. The next call to
`value_and_gradient!!` (or `value_and_jacobian!!`) may overwrite that buffer
in place, so a previously-returned reference will silently change. Copy
before holding on to a result:

```julia
val, grad = value_and_gradient!!(prepared, x1)
saved = copy(grad)                       # safe to keep
val2, grad2 = value_and_gradient!!(prepared, x2)
# `grad` may now reflect `x2`; `saved` still reflects `x1`
```

Backends that always allocate fresh output (e.g. `ForwardDiff.gradient`) do
not actually alias, but consumers should not rely on that — write to the
contract, not the implementation.

## Quick start

```@example ad
using AbstractPPL
using AbstractPPL: prepare, value_and_gradient!!
using AbstractPPL.Evaluators: Prepared, VectorEvaluator, NamedTupleEvaluator
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff

function AbstractPPL.prepare(adtype::AutoForwardDiff, f, x::AbstractVector{<:Real})
    return Prepared(adtype, VectorEvaluator(f, length(x)))
end

function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:AutoForwardDiff}, x::AbstractVector{<:Real}
)
    return (p(x), ForwardDiff.gradient(p.evaluator.f, x))
end

mvnormal_logp(x) = -0.5 * sum(abs2, x)  # standard normal log density (up to constant)
prepared = prepare(AutoForwardDiff(), mvnormal_logp, zeros(3))
value_and_gradient!!(prepared, [1.0, 2.0, 3.0])
```

## Two input styles

### Vector inputs

When the callable accepts a flat vector, pass a sample vector whose length
matches the expected input:

```@example ad
prepared([1.0, 2.0, 3.0])
```

For vector-valued callables, use `value_and_jacobian!!`. The returned Jacobian
has shape `(length(value), length(x))`:

```@example ad
using AbstractPPL: value_and_jacobian!!

vecfun(x) = [x[1] * x[2], x[2] + x[3]]

function AbstractPPL.value_and_jacobian!!(
    p::Prepared{<:AutoForwardDiff}, x::AbstractVector{<:Real}
)
    return (p(x), ForwardDiff.jacobian(p.evaluator.f, x))
end

prepared_vec = prepare(AutoForwardDiff(), vecfun, zeros(3))
value_and_jacobian!!(prepared_vec, [2.0, 3.0, 4.0])
```

### NamedTuple inputs

When the callable accepts a `NamedTuple`, pass a sample `NamedTuple` whose
field names and value types match the expected input. The prototype's leaves
must be `Real`, `Complex`, `AbstractArray` (recursively), `Tuple`, or
`NamedTuple` — the same structural model used by `flatten_to!!` /
`unflatten_to!!`. An extension can define a `prepare` overload that wraps the
function in a `NamedTupleEvaluator`:

```@example ad
function AbstractPPL.prepare(adtype::AutoForwardDiff, f, values::NamedTuple)
    return Prepared(adtype, NamedTupleEvaluator(f, values))
end

ntfun(v::NamedTuple) = v.a^2 + sum(abs2, v.b)
prepared_nt = prepare(AutoForwardDiff(), ntfun, (a=0.0, b=zeros(2)))
prepared_nt((a=1.0, b=[2.0, 3.0]))
```

## AD backends

Automatic differentiation packages extend the interface by implementing
`value_and_gradient!!` and `value_and_jacobian!!` for specific cache types
stored in `prepared.cache`:

```julia
prepared = prepare(adtype, problem, prototype)  # returns Prepared{AD,E,Cache}
value_and_gradient!!(prepared, x)               # may return aliased cache buffer
value_and_jacobian!!(prepared, x)
```

`Prepared` has three fields: `adtype`, `evaluator` (the user-facing callable),
and `cache` (backend-specific pre-allocated state such as ForwardDiff configs or
Mooncake tapes). Backend extensions dispatch on the cache type:

```julia
function AbstractPPL.prepare(
    adtype::MyADType, problem, x::AbstractVector{<:Real}; check_dims::Bool=true
)
    f = ...        # extract callable from problem
    cache = MyCache(f, x)
    return Prepared(adtype, VectorEvaluator{check_dims}(f, length(x)), cache)
end

function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:AbstractADType,<:VectorEvaluator,<:MyCache}, x::AbstractVector{<:Real}
)
    # use p.cache to avoid allocations
    return ...
end
```

Pass `check_dims=false` in your `prepare` implementation to construct a
`VectorEvaluator{false}`, which skips the per-call length check. This is an
opt-in trust mode — the caller takes responsibility for `length(x)`. The
typical use is inside a backend's `value_and_gradient!!`, where the AD
library invokes the inner callable many times with same-length dual arrays
derived from a single user-supplied `x`; re-validating on each invocation
would be redundant work in the hot path.

## Without an AD backend

The two-argument form `prepare(problem, x)` is available without any AD
package. It returns the callable unchanged by default, so the caller doesn't
need to know whether an AD backend is loaded — the same `prepare(...)` call
works either way, and downstream code that only needs primal evaluation
(e.g. log-density only, no gradient) can accept the result uniformly:

```@example ad
sumsimple(x) = sum(x)
p = prepare(sumsimple, zeros(3))
p([1.0, 2.0, 3.0])
```

## API reference

```@docs
AbstractPPL.prepare
AbstractPPL.value_and_gradient!!
AbstractPPL.value_and_jacobian!!
```
