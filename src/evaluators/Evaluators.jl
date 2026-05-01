module Evaluators

using ADTypes: AbstractADType
import ..evaluate!!

include("utils.jl")

"""
    Prepared{AD<:AbstractADType,E,C}(adtype, evaluator, cache)
    Prepared(adtype, evaluator)   # cache defaults to `nothing`

AD-prepared evaluator parameterised by backend type `AD`.

- `adtype` — the backend, used for dispatch.
- `evaluator` — the user-facing callable (typically a `VectorEvaluator` or
  `NamedTupleEvaluator`); forwarded on `p(x)`.
- `cache` — backend-specific pre-allocated state (ForwardDiff configs, Mooncake
  caches, DifferentiationInterface preps, etc.). `Nothing` when the backend requires
  no cached state.

Extension packages implement `value_and_gradient!!` (and optionally
`value_and_jacobian!!`) by specialising on the `cache` type:

```julia
function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:AbstractADType, <:VectorEvaluator, <:MyCache}, x::AbstractVector
)
    ...
end
```
"""
struct Prepared{AD<:AbstractADType,E,C}
    adtype::AD
    evaluator::E
    cache::C
end

Prepared(adtype::AbstractADType, evaluator) = Prepared(adtype, evaluator, nothing)

(p::Prepared)(x) = p.evaluator(x)

"""
    prepare(problem, values::NamedTuple; check_dims::Bool=true)
    prepare(problem, x::AbstractVector{<:Real}; check_dims::Bool=true)
    prepare(adtype, problem, x::AbstractVector{<:Real}; check_dims::Bool=true)

Prepare a callable evaluator for `problem`.

Use the two-argument form with a `NamedTuple` when the evaluator works with
named inputs, or with a vector when it works with vector inputs. The
three-argument form, contributed by AD-backend extensions, additionally
prepares gradient or jacobian machinery for vector inputs.

`check_dims` (default `true`) is forwarded to the evaluator constructor by
AD-backend extensions (three-argument form). Pass `check_dims=false` to skip
per-call shape validation, e.g. when the AD backend already guarantees the
input shape. The two-argument stubs ignore this keyword.
"""
function prepare end

# Downstream packages (e.g. DynamicPPL) pass already-callable objects,
# so the safe default is to return them unchanged.
prepare(problem, values::NamedTuple; check_dims::Bool=true) = problem
prepare(problem, x::AbstractVector{<:Real}; check_dims::Bool=true) = problem

"""
    value_and_gradient!!(prepared, x::AbstractVector{<:Real})

Return `(value, gradient)` for a scalar-valued evaluator, potentially reusing
internal cache buffers of `prepared`. The returned gradient may alias
`prepared`'s internal storage; copy if you need to retain it past the next call.
"""
function value_and_gradient!! end

"""
    value_and_jacobian!!(prepared, x::AbstractVector{<:Real})

Return `(value::AbstractVector, jacobian::AbstractMatrix)` for a vector-valued
evaluator, potentially reusing internal cache buffers. The returned arrays may
alias `prepared`'s internal storage; copy if needed.
The Jacobian has shape `(length(value), length(x))`.
"""
function value_and_jacobian!! end

"""
    VectorEvaluator{CheckInput}(f, dim)
    VectorEvaluator(f, dim)  # equivalent to `VectorEvaluator{true}(f, dim)`

Evaluator shape for scalar functions of a vector input. Part of the extension
author API; end users interact with the wrapping `Prepared` instead.

`CheckInput` controls whether each call validates the input length. The default
(`true`) is the safe shape exposed via `prepared(x)`. Pass `CheckInput=false`
(via `check_dims=false` in `prepare`) for the callable handed to AD libraries,
where input shape is already guaranteed and the runtime check would persist in
the dual/shadow hot path.

A bare `VectorEvaluator` is *not* differentiable; gradient capability is the
contract of the wrapping `Prepared` returned by `prepare(adtype, ...)`.
"""
struct VectorEvaluator{CheckInput,F}
    f::F
    dim::Int
    function VectorEvaluator{CheckInput}(f::F, dim::Int) where {CheckInput,F}
        CheckInput isa Bool || throw(ArgumentError("`CheckInput` must be a Bool."))
        dim >= 0 || throw(ArgumentError("`dim` must be non-negative, got $dim."))
        return new{CheckInput,F}(f, dim)
    end
end

VectorEvaluator(f, dim::Int) = VectorEvaluator{true}(f, dim)

"""
    NamedTupleEvaluator{CheckInput}(f, inputspec)
    NamedTupleEvaluator(f, inputspec)  # equivalent to `NamedTupleEvaluator{true}(f, inputspec)`

Evaluator shape for functions of a `NamedTuple` input with a stable prototype.
Part of the extension author API; end users interact with the wrapping `Prepared`.

`CheckInput` controls whether each call validates that the input `NamedTuple`
has the same type as the prototype captured during preparation.
"""
struct NamedTupleEvaluator{CheckInput,F,P<:NamedTuple}
    f::F
    inputspec::P
    function NamedTupleEvaluator{CheckInput}(
        f::F, inputspec::P
    ) where {CheckInput,F,P<:NamedTuple}
        CheckInput isa Bool || throw(ArgumentError("`CheckInput` must be a Bool."))
        return new{CheckInput,F,P}(f, inputspec)
    end
end

NamedTupleEvaluator(f, inputspec::NamedTuple) = NamedTupleEvaluator{true}(f, inputspec)

function (e::VectorEvaluator{true})(x::AbstractVector)
    length(x) == e.dim || throw(
        DimensionMismatch(
            "Expected a vector of length $(e.dim), but got length $(length(x))."
        ),
    )
    return e.f(x)
end

(e::VectorEvaluator{false})(x::AbstractVector) = e.f(x)

function (e::NamedTupleEvaluator{true})(values::NamedTuple)
    _assert_namedtuple_shape(e, values)
    return e.f(values)
end
(e::NamedTupleEvaluator{false})(values::NamedTuple) = e.f(values)

# Reject integer vectors with a clear error rather than letting them flow into
# AD backends (which usually fail confusingly). Split per `CheckInput` to avoid
# an ambiguity with the `(::VectorEvaluator{true})(::AbstractVector)` method above.
function _reject_integer_input(::VectorEvaluator, x)
    throw(
        ArgumentError(
            "VectorEvaluator requires a vector of floating-point values, but received an `$(typeof(x))`. Convert to a floating-point vector (e.g. `Float64.(x)`) before calling.",
        ),
    )
end
(e::VectorEvaluator{true})(x::AbstractVector{<:Integer}) = _reject_integer_input(e, x)
(e::VectorEvaluator{false})(x::AbstractVector{<:Integer}) = _reject_integer_input(e, x)

"""
    _assert_namedtuple_shape(e::NamedTupleEvaluator, values)

Throw `ArgumentError` unless `values` has the same type as the prototype captured
during preparation. No-op when `e` was constructed with `CheckInput=false`.
"""
function _assert_namedtuple_shape(e::NamedTupleEvaluator{true}, values)
    typeof(values) === typeof(e.inputspec) || throw(
        ArgumentError(
            "Expected the same NamedTuple structure that was used to prepare this evaluator.",
        ),
    )
    return nothing
end
_assert_namedtuple_shape(::NamedTupleEvaluator{false}, _) = nothing

# Output-shape assertions for AD-backend extensions to share. Centralised here
# so each backend's `value_and_gradient!!` / `value_and_jacobian!!` produces
# the same error message rather than rolling its own.
function _assert_jacobian_output(y)
    y isa AbstractVector || throw(
        ArgumentError(
            "`value_and_jacobian!!` requires the prepared function to return an AbstractVector; got $(typeof(y)).",
        ),
    )
    return nothing
end

function _assert_supported_output(y)
    (y isa Number || y isa AbstractVector) || throw(
        ArgumentError(
            "A prepared AD evaluator must return a scalar or AbstractVector; got $(typeof(y)).",
        ),
    )
    return nothing
end

# Make prepared evaluators usable through the same `evaluate!!` API as models.
evaluate!!(p::Prepared, x) = p(x)
evaluate!!(e::VectorEvaluator, x) = e(x)
evaluate!!(e::NamedTupleEvaluator, x) = e(x)

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, args, kwargs
        # `args` are argument types, not values (see `Base.Experimental.show_error_hints`).
        if exc.f === prepare && length(args) >= 1 && args[1] <: AbstractADType
            print(
                io,
                "\nCalling `prepare` with an AD backend requires loading the corresponding extension (e.g., `using DifferentiationInterface`).",
            )
        end
    end
end

end # module
