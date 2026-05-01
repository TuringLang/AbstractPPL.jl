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

`check_dims` (default `true`) controls whether the returned evaluator validates
the input shape on each call. Pass `check_dims=false` to skip the per-call
check, e.g. inside an AD backend's hot path where the input shape is already
guaranteed.
"""
function prepare end

# Default: wrap the callable in the appropriate evaluator so per-call shape
# checks fire even without a backend-specific `prepare` method. Downstream
# packages (e.g. DynamicPPL) override these for their problem types.
function prepare(problem, values::NamedTuple; check_dims::Bool=true)
    return NamedTupleEvaluator{check_dims}(problem, values)
end
function prepare(problem, x::AbstractVector{<:Real}; check_dims::Bool=true)
    return VectorEvaluator{check_dims}(problem, length(x))
end

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

The `inputspec` prototype's leaves must be one of:

- `Real` or `Complex` (scalar)
- `AbstractArray` whose elements are themselves supported leaves
- `Tuple` or `NamedTuple` recursively containing supported leaves

This matches the structural model used by [`flatten_to!!`](@ref) /
[`unflatten_to!!`](@ref). Other leaf types (e.g. `String`, `Symbol`, custom
structs) trigger an `ArgumentError` from the per-call shape check.

`CheckInput` controls whether each call validates that the input `NamedTuple`
matches the prototype's `typeof` and per-leaf array `size`.
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

# Reject integer vectors with a clear error rather than letting them flow into
# AD backends (which usually fail confusingly). The `T <: Integer` branch is
# resolved at compile time, so non-integer inputs pay nothing.
function _reject_integer_input(x)
    throw(
        ArgumentError(
            "VectorEvaluator requires a vector of floating-point values, but received an `$(typeof(x))`. Convert to a floating-point vector (e.g. `Float64.(x)`) before calling.",
        ),
    )
end

function (e::VectorEvaluator{true})(x::AbstractVector{T}) where {T}
    T <: Integer && _reject_integer_input(x)
    length(x) == e.dim || throw(
        DimensionMismatch(
            "Expected a vector of length $(e.dim), but got length $(length(x))."
        ),
    )
    return e.f(x)
end

function (e::VectorEvaluator{false})(x::AbstractVector{T}) where {T}
    T <: Integer && _reject_integer_input(x)
    return e.f(x)
end

function (e::NamedTupleEvaluator{true})(values::NamedTuple)
    _assert_namedtuple_shape(e, values)
    return e.f(values)
end
(e::NamedTupleEvaluator{false})(values::NamedTuple) = e.f(values)

"""
    _assert_namedtuple_shape(e::NamedTupleEvaluator, values)

Throw `ArgumentError` unless `values` has the same type as the prototype captured
during preparation, including matching `size` for any nested `AbstractArray`
leaves. Also throws if the prototype contains a leaf type outside the supported
set (`Real`, `Complex`, `AbstractArray`, `Tuple`, `NamedTuple`). No-op when `e`
was constructed with `CheckInput=false`.
"""
function _assert_namedtuple_shape(e::NamedTupleEvaluator{true}, values)
    typeof(values) === typeof(e.inputspec) || throw(
        ArgumentError(
            "Expected the same NamedTuple structure that was used to prepare this evaluator.",
        ),
    )
    _shapes_match(values, e.inputspec) || throw(
        ArgumentError(
            "Nested array shape differs from the prototype captured during preparation."
        ),
    )
    return nothing
end
_assert_namedtuple_shape(::NamedTupleEvaluator{false}, _) = nothing

# Complements the `typeof` check above: same-typed arrays can differ in `size`.
# Arrays with non-`Real`/`Complex` eltype are walked element-wise to catch
# inner mismatches. Unknown leaves throw, mirroring the supported-leaves
# contract of the flatten/unflatten utilities in `utils.jl`.
#
# `Tuple` recursion uses `first`/`Base.tail` rather than a `zip` loop so each
# leaf call sees concrete element types — same idiom as `_unflatten`.
_shapes_match(::Union{Real,Complex}, ::Union{Real,Complex}) = true
function _shapes_match(a::AbstractArray, b::AbstractArray)
    size(a) == size(b) || return false
    eltype(a) <: Union{Real,Complex} && return true
    for (ai, bi) in zip(a, b)
        _shapes_match(ai, bi) || return false
    end
    return true
end
_shapes_match(::Tuple{}, ::Tuple{}) = true
function _shapes_match(a::Tuple, b::Tuple)
    _shapes_match(first(a), first(b)) || return false
    return _shapes_match(Base.tail(a), Base.tail(b))
end
_shapes_match(a::NamedTuple, b::NamedTuple) = _shapes_match(values(a), values(b))
function _shapes_match(a, _)
    throw(
        ArgumentError(
            "Cannot validate shape for prototype leaf of type `$(typeof(a))`. Supported leaves are `Real`, `Complex`, `AbstractArray`, `Tuple`, and `NamedTuple`.",
        ),
    )
end

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
