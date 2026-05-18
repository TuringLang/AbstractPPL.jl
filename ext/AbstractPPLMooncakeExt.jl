module AbstractPPLMooncakeExt

using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators:
    Evaluators, Prepared, VectorEvaluator, NamedTupleEvaluator, _ad_output_arity
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake

const _MooncakeAD = Union{AutoMooncake,AutoMooncakeForward}

# `NamedTupleEvaluator` is the callable on the NamedTuple path; `NoTangent`
# stops Mooncake from deriving a `Tangent{NamedTuple{...}}` for its fields
# on every backward pass. `VectorEvaluator` is no longer passed to Mooncake
# after the raw-target merge; the override is kept as a defensive guard.
Mooncake.tangent_type(::Type{<:VectorEvaluator}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:NamedTupleEvaluator}) = Mooncake.NoTangent

# Type parameters:
#
#   * `A::Symbol` — output arity, `:scalar` or `:vector`. Drives the
#                   gradient/jacobian dispatch and the arity-mismatch errors.
#   * `C`         — the underlying Mooncake cache, or `Nothing` for the
#                   empty-input shortcut.
struct MooncakeCache{A,C}
    cache::C
end
MooncakeCache{A}(cache::C) where {A,C} = MooncakeCache{A,C}(cache)

_mooncake_config(adtype) = adtype.config === nothing ? Mooncake.Config() : adtype.config

function _mooncake_gradient_cache(::AutoMooncake, f, x; config)
    return Mooncake.prepare_gradient_cache(f, x; config)
end
function _mooncake_gradient_cache(::AutoMooncakeForward, f, x; config)
    return Mooncake.prepare_derivative_cache(f, x; config)
end

# Vector-scalar overloads — splat `context` into the underlying Mooncake
# prep call (empty tuple is a no-op).
function _mooncake_gradient_cache(::AutoMooncake, f, x, context::Tuple; config)
    return Mooncake.prepare_gradient_cache(f, x, context...; config)
end
function _mooncake_gradient_cache(::AutoMooncakeForward, f, x, context::Tuple; config)
    return Mooncake.prepare_derivative_cache(f, x, context...; config)
end

function _mooncake_jacobian_cache(::AutoMooncake, f, x; config)
    return Mooncake.prepare_pullback_cache(f, x; config)
end
function _mooncake_jacobian_cache(::AutoMooncakeForward, f, x; config)
    return Mooncake.prepare_derivative_cache(f, x; config)
end

function AbstractPPL.prepare(
    adtype::_MooncakeAD, problem, values::NamedTuple; check_dims::Bool=true
)
    evaluator = AbstractPPL.prepare(problem, values; check_dims)::NamedTupleEvaluator
    config = _mooncake_config(adtype)
    cache = _mooncake_gradient_cache(adtype, evaluator, values; config)
    return Prepared(adtype, evaluator, cache)
end

"""
    prepare(adtype::AutoMooncake, problem, x; check_dims=true, context::Tuple=())
    prepare(adtype::AutoMooncakeForward, problem, x; check_dims=true, context::Tuple=())

Prepare a Mooncake gradient/Jacobian evaluator for a dense vector input.

Non-`DenseVector` inputs (views, strided slices) are rejected: Mooncake
assumes a contiguous primal and otherwise returns shape-incorrect tangents
on reverse mode and crashes on forward/Jacobian paths.

`context` follows the base `prepare` contract — the prepared evaluator
computes `problem(x, context...)` with AD differentiating only `x`. One
Mooncake-specific restriction: vector-valued problems require `context=()`.

Empty input (`length(x) == 0`) is supported with any `context`; Mooncake
builds no tape for zero-length `x`, so the prepared evaluator's AD entry
short-circuits to `(problem(x, context...), eltype(x)[])` without invoking
Mooncake.
"""
function AbstractPPL.prepare(
    adtype::_MooncakeAD,
    problem,
    x::AbstractVector{<:Real};
    check_dims::Bool=true,
    context::Tuple=(),
)
    x isa DenseVector || throw(
        ArgumentError(
            "AutoMooncake / AutoMooncakeForward require a dense vector input " *
            "(e.g. `Vector{<:Real}`); got $(typeof(x)). Wrap non-dense inputs " *
            "(views, strided slices) with `collect` before calling `prepare`.",
        ),
    )
    evaluator = AbstractPPL.prepare(problem, x; check_dims, context)::VectorEvaluator
    arity = _ad_output_arity(evaluator(x))
    config = _mooncake_config(adtype)
    if !isempty(evaluator.context) && arity !== :scalar
        throw(
            ArgumentError(
                "Non-empty `context` is only supported for scalar-valued problems."
            ),
        )
    end
    # Mooncake builds no tape for length-zero `x`; tag with `Nothing` so the
    # empty-input methods below shortcut without invoking Mooncake. Empty `x`
    # with non-empty context also routes here — the hot-path shortcut just
    # calls `p.evaluator(x)` which already does `f([], context...)`.
    length(x) == 0 && return Prepared(adtype, evaluator, MooncakeCache{arity}(nothing))
    # Compile the tape on the evaluator's `f` and `context` (not the raw
    # `problem` passed in): a downstream override of structural `prepare`
    # may return a `VectorEvaluator` whose `.f`/`.context` differ from the
    # caller-supplied `problem`/`context`. The hot path uses `evaluator.f`
    # / `evaluator.context`, so the cache must agree.
    cache = if arity === :scalar
        _mooncake_gradient_cache(adtype, evaluator.f, x, evaluator.context; config)
    else
        _mooncake_jacobian_cache(adtype, evaluator.f, x; config)
    end
    return Prepared(adtype, evaluator, MooncakeCache{arity}(cache))
end

# Input-shape validation is delegated to the AD backend: Mooncake catches
# top-level NamedTuple-type mismatches, and the inner
# `NamedTupleEvaluator{CheckInput}` callable catches nested-array size
# mismatches (gated by `check_dims`). Running `_assert_namedtuple_shape`
# again here would duplicate the second check on every AD call.
# (`∂f` is `NoTangent` thanks to the `tangent_type` overload above.)
@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:_MooncakeAD,<:NamedTupleEvaluator}, values::NamedTuple
)
    val, (_, grad) = Mooncake.value_and_gradient!!(p.cache, p.evaluator, values)
    return (val, grad)
end

# Empty-input shortcut. `MooncakeCache{:scalar,Nothing}` is strictly more
# specific than `MooncakeCache{:scalar}` on `C`, so dispatch unambiguously
# selects this method over the general scalar-gradient hot path below for
# zero-length `x`.
@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:scalar,Nothing}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    return (p.evaluator(x), T[])
end

# Reverse-mode `Mooncake.Cache` needs `args_to_zero` to mark `x` as the lone
# active input (`false` on `f`, `true` on `x`, `false` on each context value).
# Forward-mode `ForwardCache` derives activity from its seeded argument and
# rejects the kwarg. Dispatching on the AD type keeps each call mode-specific
# without a runtime branch.
@inline _mooncake_value_and_gradient(
    ::AutoMooncake, cache, f::F, x, context::Tuple
) where {F} = Mooncake.value_and_gradient!!(
    cache, f, x, context...; args_to_zero=(false, true, map(_ -> false, context)...)
)
@inline _mooncake_value_and_gradient(
    ::AutoMooncakeForward, cache, f::F, x, context::Tuple
) where {F} = Mooncake.value_and_gradient!!(cache, f, x, context...)

# Scalar-gradient hot path. Empty `context` collapses the splat and reduces
# `args_to_zero` to `(false, true)`. `tangents[2]` is the `x`-gradient — the
# trailing entries (one per context value) are zeroed and discarded.
@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:scalar}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    e = p.evaluator
    val, tangents = _mooncake_value_and_gradient(p.adtype, p.cache.cache, e.f, x, e.context)
    return (val, tangents[2])
end

# Arity-mismatch errors as dedicated methods so dispatch on
# `MooncakeCache{:scalar}` vs `{:vector}` resolves at compile time instead of
# a runtime check on the cache contents.
@inline function AbstractPPL.value_and_gradient!!(
    ::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:vector}},
    ::AbstractVector{<:Real},
)
    return Evaluators._throw_gradient_needs_scalar()
end

@inline function AbstractPPL.value_and_jacobian!!(
    ::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:scalar}},
    ::AbstractVector{<:Real},
)
    return Evaluators._throw_jacobian_needs_vector()
end

# Empty-input jacobian shortcut. Same `Nothing` specificity trick as the
# scalar case above; skips Mooncake entirely.
@inline function AbstractPPL.value_and_jacobian!!(
    p::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:vector,Nothing}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    val = p.evaluator(x)
    return (val, similar(x, length(val), 0))
end

@inline function AbstractPPL.value_and_jacobian!!(
    p::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:vector}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    # Vector arity rejects non-empty `context` at prepare time, so the tape
    # is compiled on `problem(x)` and there is no splat or `args_to_zero` to
    # propagate. Mooncake's `value_and_jacobian!!` returns `(val, jac)`
    # directly with `x` as the only active argument.
    return Mooncake.value_and_jacobian!!(p.cache.cache, p.evaluator.f, x)
end

end # module
