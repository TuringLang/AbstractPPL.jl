module Evaluators

using ADTypes: AbstractADType
import ..evaluate!!

include("utils.jl")

"""
    Prepared{AD<:AbstractADType,E,C,Order}(adtype, evaluator, cache)
    Prepared(adtype, evaluator, cache, Val(Order))
    Prepared(adtype, evaluator, cache)          # defaults `Order` to 1
    Prepared(adtype, evaluator)                 # cache defaults to `nothing`

AD-prepared evaluator parameterised by backend type `AD` and derivative order
`Order` (`1` for gradient/jacobian, `2` for Hessian). Retrieve `Order` via
[`order`](@ref).

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
struct Prepared{AD<:AbstractADType,E,C,Order}
    adtype::AD
    evaluator::E
    cache::C
    function Prepared{AD,E,C,Order}(
        adtype, evaluator, cache
    ) where {AD<:AbstractADType,E,C,Order}
        return new{AD,E,C,Order}(adtype, evaluator, cache)
    end
end

function Prepared(
    adtype::AD, evaluator::E, cache::C, ::Val{Order}
) where {AD<:AbstractADType,E,C,Order}
    return Prepared{AD,E,C,Order}(adtype, evaluator, cache)
end
function Prepared(adtype::AbstractADType, evaluator, cache)
    return Prepared(adtype, evaluator, cache, Val(1))
end
Prepared(adtype::AbstractADType, evaluator) = Prepared(adtype, evaluator, nothing, Val(1))

(p::Prepared)(x) = p.evaluator(x)

"""
    order(p::Prepared)

Return the derivative order `p` was prepared for (`1` for gradient/jacobian,
`2` for Hessian). Type-stable — folds to the `Order` type parameter at compile
time.
"""
order(::Prepared{<:Any,<:Any,<:Any,O}) where {O} = O

"""
    prepare(problem, values::NamedTuple; check_dims::Bool=true)
    prepare(problem, x::AbstractVector{<:Real}; check_dims::Bool=true, context::Tuple=())
    prepare(adtype, problem, x::AbstractVector{<:Real}; check_dims::Bool=true, context::Tuple=(), order::Int=1)

Prepare a callable evaluator for `problem`.

Use the two-argument form with a `NamedTuple` when the evaluator works with
named inputs, or with a vector when it works with vector inputs. The
three-argument form, contributed by AD-backend extensions, additionally
prepares gradient, jacobian, or Hessian machinery for vector inputs.

`check_dims` (default `true`) controls whether the returned evaluator validates
the input shape on each call. Pass `check_dims=false` to skip the per-call
check, e.g. inside an AD backend's hot path where the input shape is already
guaranteed.

The vector-input forms accept a `context::Tuple` of constant arguments threaded
through to `problem`: the prepared evaluator computes `problem(x, context...)`,
and AD backends differentiate only with respect to `x`. `context=()` (the
default) preserves the unary `problem(x)` contract.

`order` selects the derivative order to prepare for on the AD-aware form. The
default `order=1` prepares gradient (scalar output) or jacobian (vector output)
machinery. `order=2` prepares Hessian machinery via `value_gradient_and_hessian!!`
and requires `problem` to be scalar-valued — vector-valued problems will throw
during preparation.

The three-argument AD-aware form may invoke `problem` once during preparation
to detect output arity (scalar vs vector) and select the appropriate
derivative machinery. Avoid `prepare` calls when `problem` has side effects
that should fire only on user-driven evaluations.
"""
function prepare end

# Default: wrap the callable in the appropriate evaluator so per-call shape
# checks fire even without a backend-specific `prepare` method. Downstream
# packages (e.g. DynamicPPL) override these for their problem types.
function prepare(problem, values::NamedTuple; check_dims::Bool=true)
    return NamedTupleEvaluator{check_dims}(problem, values)
end
function prepare(
    problem, x::AbstractVector{<:Real}; check_dims::Bool=true, context::Tuple=()
)
    return VectorEvaluator{check_dims}(problem, length(x), context)
end

"""
    value_and_gradient!!(prepared, x::AbstractVector{<:Real}; context=nothing)

Return `(value, gradient)` for a scalar-valued evaluator, potentially reusing
internal cache buffers of `prepared`. The returned gradient may alias
`prepared`'s internal storage; copy if you need to retain it past the next call.

By default the `context` frozen at `prepare` is used. Pass a `Tuple` as
`context` to override it for a single call without re-preparing — it must match
the prepared context's element types and shapes, since the prepared cache is
keyed on types. The override is per-call and does not mutate the frozen context.
Compiled-tape ReverseDiff (`AutoReverseDiff(; compile=true)`) bakes the context
into its tape and throws if an override is supplied for a non-empty input.
Empty input (`length(x) == 0`) runs no derivative machinery — the value is
computed directly — so a `context` override is always accepted there.
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
    value_gradient_and_hessian!!(prepared, x::AbstractVector{<:Real}; context=nothing)

Return `(value, gradient::AbstractVector, hessian::AbstractMatrix)` for a
scalar-valued evaluator prepared with `order=2`, potentially reusing internal
cache buffers. The returned gradient and Hessian may alias `prepared`'s
internal storage; copy if you need to retain them past the next call.
The Hessian has shape `(length(x), length(x))`.

`context` may override the context frozen at `prepare` for a single call, under
the same contract as [`value_and_gradient!!`](@ref), except that on a non-empty
input two backends throw for the Hessian and require re-`prepare` instead:
compiled-tape ReverseDiff (`AutoReverseDiff(; compile=true)`), which bakes the
context into its tape, and Mooncake, whose Hessian cache binds its target by
object identity. As for the gradient, empty input runs no machinery, so an
override is always accepted there.
"""
function value_gradient_and_hessian!! end

"""
    VectorEvaluator{CheckInput}(f, dim, context::Tuple=())
    VectorEvaluator(f, dim, context::Tuple=())  # equivalent to `VectorEvaluator{true}(f, dim, context)`

Evaluator shape for scalar functions of a vector input. Part of the extension
author API; end users interact with the wrapping `Prepared` instead.

`CheckInput` controls whether each call validates the input length. The default
(`true`) is the safe shape exposed via `prepared(x)`. Pass `CheckInput=false`
(via `check_dims=false` in `prepare`) for the callable handed to AD libraries,
where input shape is already guaranteed and the runtime check would persist in
the dual/shadow hot path.

`context` is a tuple of constant arguments threaded through to `f`:
`evaluator(x)` computes `f(x, context...)`. AD backends treat every value in
`context` as inactive and differentiate only with respect to `x`. The default
empty tuple keeps the unary `f(x)` contract.

A bare `VectorEvaluator` is *not* differentiable; gradient capability is the
contract of the wrapping `Prepared` returned by `prepare(adtype, ...)`.
"""
struct VectorEvaluator{CheckInput,F,C<:Tuple}
    f::F
    dim::Int
    context::C
    function VectorEvaluator{CheckInput}(
        f::F, dim::Int, context::C=()
    ) where {CheckInput,F,C<:Tuple}
        CheckInput isa Bool || throw(ArgumentError("`CheckInput` must be a Bool."))
        dim >= 0 || throw(ArgumentError("`dim` must be non-negative, got $dim."))
        return new{CheckInput,F,C}(f, dim, context)
    end
end

VectorEvaluator(f, dim::Int, context::Tuple=()) = VectorEvaluator{true}(f, dim, context)

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
matches the prototype's `typeof` and per-leaf array `size`. The default (`true`)
is the safe shape exposed via `prepared(x)`. Pass `CheckInput=false` (via
`check_dims=false` in `prepare`) for the callable handed to AD libraries: the
prototype's `typeof` is captured at preparation time using the original element
types, so a `CheckInput=true` evaluator will reject inputs whose leaves are
dual/shadow numbers (or any other widened element type) even when the structure
is otherwise correct.
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
# AD backends (which usually fail confusingly). `T <: Integer` resolves at
# compile time, so the AD hot path (Float/dual `T`) elides the branch entirely.
function _reject_integer_input(x)
    throw(
        ArgumentError(
            "VectorEvaluator requires a vector of floating-point values, but received an `$(typeof(x))`. Convert to a floating-point vector (e.g. `Float64.(x)`) before calling.",
        ),
    )
end

function _check_vector_length(dim::Int, x)
    length(x) == dim || throw(
        DimensionMismatch("Expected a vector of length $dim, but got length $(length(x))."),
    )
    return nothing
end

# Shared input validation for AD-backend `value_and_{gradient,jacobian}!!` entry
# points. Same compile-time `T <: Integer` elision as the `VectorEvaluator` body.
# Gated by `CheckInput`: the `{false}` overload is a no-op so the AD hot path
# pays nothing when the caller has already validated the input (e.g. via
# `prepare(...; check_dims=false)`).
function _check_ad_input(e::VectorEvaluator{true}, x::AbstractVector{T}) where {T}
    T <: Integer && _reject_integer_input(x)
    _check_vector_length(e.dim, x)
    return nothing
end
_check_ad_input(::VectorEvaluator{false}, ::AbstractVector) = nothing

# Primal value under a possibly-overridden call-time context, shared by the
# AD-backend extensions' empty-input shortcuts (which bypass the backend).
# `nothing` keeps the frozen context via the evaluator (honouring its
# `CheckInput`); a `Tuple` calls `f` with the override directly (the per-call
# shape check having already run via `_check_ad_input` at the entry point).
@inline _evaluate_with_context(e::VectorEvaluator, x, ::Nothing) = e(x)
@inline _evaluate_with_context(e::VectorEvaluator, x, context::Tuple) = e.f(x, context...)

# Resolve a call-time `context` override into the context tuple a backend builds
# its AD target from: `nothing` (the default) keeps the context frozen at
# `prepare`; a `Tuple` replaces it for this call (issue #167). Shared by the
# AD-backend override paths so the `nothing`/`Tuple` dispatch lives in one place;
# each backend wraps the result in its own target (`Constant`s, `_ADTarget`, a
# `Fix2` evaluator). An override must match the frozen context's element types
# and shapes, since the prepared cache is keyed on types.
@inline _resolve_context(e::VectorEvaluator, ::Nothing) = e.context
@inline _resolve_context(::VectorEvaluator, context::Tuple) = context

# Both bodies rely on `T <: Integer` being a static check so the AD hot path
# (Float/dual `T`) elides the branch; the `{false}` callable additionally skips
# `_check_vector_length` since AD libraries pass length-matching dual inputs.
function (e::VectorEvaluator{true})(x::AbstractVector{T}) where {T}
    T <: Integer && _reject_integer_input(x)
    _check_vector_length(e.dim, x)
    return e.f(x, e.context...)
end

function (e::VectorEvaluator{false})(x::AbstractVector{T}) where {T}
    T <: Integer && _reject_integer_input(x)
    return e.f(x, e.context...)
end

function (e::NamedTupleEvaluator{true})(values::NamedTuple)
    _assert_namedtuple_shape(e, values)
    return e.f(values)
end
(e::NamedTupleEvaluator{false})(values::NamedTuple) = e.f(values)

"""
    _assert_namedtuple_shape(e::NamedTupleEvaluator{true}, values)

Throw `ArgumentError` unless `values` has the same type as the prototype captured
during preparation, including matching `size` for any nested `AbstractArray`
leaves. Also throws if the prototype contains a leaf type outside the supported
set (`Real`, `Complex`, `AbstractArray`, `Tuple`, `NamedTuple`).

Gated by `CheckInput`: the `{false}` overload is a no-op so AD hot paths and
other opt-out callers pay nothing.
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

# Classify the output of a probe `evaluator(x)` call into the two arities the
# AD interface supports — `:scalar` routes to gradient prep, `:vector` to
# jacobian prep. Shared by the DI and Mooncake extensions so both surface the
# same error message for unsupported output types.
function _ad_output_arity(y)
    y isa Number && return :scalar
    y isa AbstractVector && return :vector
    throw(
        ArgumentError(
            "A prepared AD evaluator must return a scalar or AbstractVector; got $(typeof(y)).",
        ),
    )
end

# Error helpers shared by the DI and Mooncake extensions; kept here so the
# `:edge` testcase regexes (`r"scalar-valued"`, `r"vector-valued"`, `r"order=2"`)
# pin a single error string instead of one per backend.
function _throw_gradient_needs_scalar()
    throw(ArgumentError("`value_and_gradient!!` requires a scalar-valued function."))
end
function _throw_jacobian_needs_vector()
    throw(ArgumentError("`value_and_jacobian!!` requires a vector-valued function."))
end
function _throw_hessian_needs_scalar()
    throw(
        ArgumentError("`value_gradient_and_hessian!!` requires a scalar-valued function.")
    )
end
function _throw_hessian_needs_order_2_prep()
    throw(
        ArgumentError(
            "`value_gradient_and_hessian!!` requires an evaluator prepared with `order=2`."
        ),
    )
end

# Validate the `order=` kwarg of `prepare(adtype, problem, x; order)`. Shared by
# the DI and Mooncake extensions so the error string is identical.
@inline _validate_ad_order(order::Int) =
    order in (1, 2) || throw(ArgumentError("`order` must be 1 or 2, got $order."))

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

# Make prepared evaluators usable through the same `evaluate!!` API as models.
evaluate!!(e::Union{Prepared,VectorEvaluator,NamedTupleEvaluator}, x) = e(x)

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, args, kwargs
        # `args` are argument types, not values (see `Base.Experimental.show_error_hints`).
        # Only fire when no extension has registered any AD-aware `prepare` method yet —
        # once a backend is loaded, the candidate list in the `MethodError` is more
        # informative than a generic "load an extension" hint.
        exc.f === prepare || return nothing
        length(args) >= 1 && args[1] <: AbstractADType || return nothing
        # `nargs` counts `self`, so `>= 4` matches the AD-aware 3-positional form.
        any(m -> m.nargs >= 4, methods(prepare)) && return nothing
        print(
            io,
            "\nCalling `prepare` with an AD backend requires loading the corresponding extension (e.g., `using DifferentiationInterface`).",
        )
    end
    # Same fire-only-when-no-backend-loaded logic as the `prepare` hint above.
    Base.Experimental.register_error_hint(MethodError) do io, exc, args, kwargs
        exc.f === value_and_gradient!! ||
            exc.f === value_and_jacobian!! ||
            exc.f === value_gradient_and_hessian!! ||
            return nothing
        isempty(methods(exc.f)) || return nothing
        print(
            io,
            "\nNo AD backend extension is loaded. Load `DifferentiationInterface` (with a backend like `ForwardDiff`) or `Mooncake` to enable gradient/jacobian/Hessian computation.",
        )
    end
end

end # module
