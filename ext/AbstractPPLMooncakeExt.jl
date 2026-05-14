module AbstractPPLMooncakeExt

using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators:
    Evaluators, Prepared, VectorEvaluator, NamedTupleEvaluator, _ad_output_arity
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake

const _MooncakeAD = Union{AutoMooncake,AutoMooncakeForward}

# Tell Mooncake that the evaluator wrappers are constants from its
# perspective: their fields hold the user's problem state, which Mooncake
# would otherwise derive a nested `Tangent{NamedTuple{f::Tangent{...}}}` for
# and walk on every backward pass. The evaluators are AbstractPPL's own
# types and only ever appear as the callable argument to Mooncake — no
# downstream caller asks for a gradient w.r.t. them.
Mooncake.tangent_type(::Type{<:VectorEvaluator}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:NamedTupleEvaluator}) = Mooncake.NoTangent

# `A` tags the evaluator's output arity (`:scalar`/`:vector`) so arity
# mismatches dispatch to a dedicated error method instead of failing inside
# Mooncake. `f`/`contexts` are `nothing` on the generic path; on the
# `raw_gradient_target` path they carry the lowered target so `CT<:Tuple`
# selects the lowered AD entry by dispatch.
struct MooncakeCache{A,C,F,CT}
    cache::C
    f::F
    contexts::CT
end
function MooncakeCache{A}(cache::C) where {A,C}
    return MooncakeCache{A,C,Nothing,Nothing}(cache, nothing, nothing)
end
function MooncakeCache{A}(cache::C, f::F, contexts::CT) where {A,C,F,CT<:Tuple}
    return MooncakeCache{A,C,F,CT}(cache, f, contexts)
end

_mooncake_config(adtype) = adtype.config === nothing ? Mooncake.Config() : adtype.config

function _mooncake_gradient_cache(::AutoMooncake, f, x; config)
    return Mooncake.prepare_gradient_cache(f, x; config)
end
function _mooncake_gradient_cache(::AutoMooncakeForward, f, x; config)
    return Mooncake.prepare_derivative_cache(f, x; config)
end

function _mooncake_jacobian_cache(::AutoMooncake, f, x; config)
    return Mooncake.prepare_pullback_cache(f, x; config)
end
function _mooncake_jacobian_cache(::AutoMooncakeForward, f, x; config)
    return Mooncake.prepare_derivative_cache(f, x; config)
end

function AbstractPPL.prepare(
    adtype::_MooncakeAD,
    problem,
    values::NamedTuple;
    check_dims::Bool=true,
    raw_gradient_target=nothing,
)
    raw_gradient_target === nothing || throw(
        ArgumentError(
            "`raw_gradient_target` is only supported on the vector `prepare` path."
        ),
    )
    evaluator = AbstractPPL.prepare(problem, values; check_dims)::NamedTupleEvaluator
    config = _mooncake_config(adtype)
    cache = _mooncake_gradient_cache(adtype, evaluator, values; config)
    return Prepared(adtype, evaluator, cache)
end

"""
    prepare(adtype::AutoMooncake, problem, x; check_dims=true, raw_gradient_target=nothing)
    prepare(adtype::AutoMooncakeForward, problem, x; check_dims=true)

Prepare a Mooncake gradient/Jacobian evaluator for a dense vector input.

Non-`DenseVector` inputs (views, strided slices) are rejected: Mooncake
assumes a contiguous primal and otherwise returns shape-incorrect tangents
on reverse mode and crashes on forward/Jacobian paths.

# `raw_gradient_target` (unsafe)

Optional reverse-mode kwarg of the form `(f, contexts::Tuple)`. When
supplied, Mooncake compiles its tape against `f(x, contexts...)` instead of
the wrapping `VectorEvaluator`, which avoids the per-call indirection
through the evaluator on the AD hot path.

This is an **unsafe escape hatch**:

  - The caller asserts `f(x, contexts...) ≡ evaluator(x)` for every `x`
    Mooncake will see — AbstractPPL does not (and cannot) verify this.
  - The AD pass calls `f(x, contexts...)` directly; the `VectorEvaluator`
    wrapper is bypassed. Input shape is still validated up front by
    `_check_ad_input` on the user-facing call.
  - The `(f, contexts)` shape is destructured directly; malformed values
    (e.g. a bare function, or `contexts` that isn't a tuple) will raise
    `MethodError`/`BoundsError` rather than a structured `ArgumentError`.

Use only when the indirection cost is measured and the equivalence is
known to hold.
"""
function AbstractPPL.prepare(
    adtype::_MooncakeAD,
    problem,
    x::AbstractVector{<:Real};
    check_dims::Bool=true,
    raw_gradient_target=nothing,
)
    x isa DenseVector || throw(
        ArgumentError(
            "AutoMooncake / AutoMooncakeForward require a dense vector input " *
            "(e.g. `Vector{<:Real}`); got $(typeof(x)). Wrap non-dense inputs " *
            "(views, strided slices) with `collect` before calling `prepare`.",
        ),
    )
    # Validate `raw_gradient_target` preconditions that don't need an arity
    # probe, so the probe `evaluator(x)` below cannot crash on user code that
    # assumes non-empty `x`.
    if raw_gradient_target !== nothing
        adtype isa AutoMooncake || throw(
            ArgumentError(
                "`raw_gradient_target` is only supported with reverse-mode `AutoMooncake`.",
            ),
        )
        length(x) > 0 ||
            throw(ArgumentError("`raw_gradient_target` is not supported for empty input."))
    end
    evaluator = AbstractPPL.prepare(problem, x; check_dims)::VectorEvaluator
    arity = _ad_output_arity(evaluator(x))
    config = _mooncake_config(adtype)
    if raw_gradient_target !== nothing
        arity === :scalar || throw(
            ArgumentError(
                "`raw_gradient_target` is only supported for scalar-valued problems."
            ),
        )
        f, contexts = raw_gradient_target
        cache = Mooncake.prepare_gradient_cache(f, x, contexts...; config)
        return Prepared(adtype, evaluator, MooncakeCache{:scalar}(cache, f, contexts))
    end
    # Mooncake builds no tape for length-zero `x`; tag with `Nothing` so the
    # empty-input methods below shortcut without invoking Mooncake.
    length(x) == 0 && return Prepared(adtype, evaluator, MooncakeCache{arity}(nothing))
    cache = if arity === :scalar
        _mooncake_gradient_cache(adtype, evaluator, x; config)
    else
        _mooncake_jacobian_cache(adtype, evaluator, x; config)
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

# Empty-input shortcut: tagged with `MooncakeCache{…,Nothing}` at prepare time
# so dispatch resolves the no-Mooncake path at compile time — no runtime
# `isnothing(cache)` branch in the hot path.
@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:scalar,Nothing}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    return (p.evaluator(x), T[])
end

@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:scalar}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    val, (_, grad) = Mooncake.value_and_gradient!!(p.cache.cache, p.evaluator, x)
    return (val, grad)
end

# Lowered raw-target gradient — `p.cache.f(x, p.cache.contexts...) ≡ p.evaluator(x)`
# by the `raw_gradient_target` contract. Mooncake's tape was compiled on the
# raw shape, sidestepping the fixed `evaluator(x)` overhead. `CT<:Tuple`
# distinguishes the lowered cache from the generic one (where `CT=Nothing`).
# `args_to_zero` is constant-folded from `c.contexts`'s arity at compile time.
@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{
        <:AutoMooncake,<:VectorEvaluator,<:MooncakeCache{:scalar,<:Any,<:Any,<:Tuple}
    },
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    c = p.cache
    val, tangents = Mooncake.value_and_gradient!!(
        c.cache,
        c.f,
        x,
        c.contexts...;
        args_to_zero=(false, true, map(_ -> false, c.contexts)...),
    )
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

# Empty-input jacobian shortcut — same compile-time dispatch trick as the
# scalar Nothing-tagged case; skips Mooncake entirely.
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
    return Mooncake.value_and_jacobian!!(p.cache.cache, p.evaluator, x)
end

end # module
