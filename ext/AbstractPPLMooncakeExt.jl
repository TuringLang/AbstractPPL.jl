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

# Tag a Mooncake cache with the prepared evaluator's output arity (`:scalar`
# or `:vector`) so `value_and_gradient!!` / `value_and_jacobian!!` can raise
# helpful arity-mismatch errors instead of failing inside Mooncake.
struct MooncakeCache{A,C}
    cache::C
end
MooncakeCache{A}(cache::C) where {A,C} = MooncakeCache{A,C}(cache)

# Opt-in lowered-target cache for reverse-mode `AutoMooncake`: callers who
# know a raw `f(x, contexts...)` equivalent to `problem(x)` can hand it in
# via `raw_gradient_target=(f, contexts)`. Mooncake then compiles a tape on
# `(f, x, contexts...)` rather than the generic `evaluator(x)` shape — the
# inactive `contexts` ride along as plain positional args with
# `args_to_zero=false`. `prepared(x)` still calls `problem(x)`; only the AD
# entry point uses the lowered cache.
struct MooncakeLoweredCache{C,F,CT<:Tuple,AZ<:Tuple}
    cache::C
    f::F
    contexts::CT
    args_to_zero::AZ
end

_mooncake_config(adtype) = adtype.config === nothing ? Mooncake.Config() : adtype.config

# `value_and_gradient!!` accepts either a reverse-mode gradient cache
# (AutoMooncake) or a forward-mode derivative cache (AutoMooncakeForward).
function _mooncake_gradient_cache(::AutoMooncake, f, x; config)
    return Mooncake.prepare_gradient_cache(f, x; config)
end
function _mooncake_gradient_cache(::AutoMooncakeForward, f, x; config)
    return Mooncake.prepare_derivative_cache(f, x; config)
end

# `value_and_jacobian!!`: reverse mode wants a pullback cache, forward mode
# wants a derivative cache.
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

function AbstractPPL.prepare(
    adtype::_MooncakeAD,
    problem,
    x::AbstractVector{<:Real};
    check_dims::Bool=true,
    raw_gradient_target=nothing,
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
        args_to_zero = (false, true, map(_ -> false, contexts)...)
        return Prepared(
            adtype, evaluator, MooncakeLoweredCache(cache, f, contexts, args_to_zero)
        )
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

# `Mooncake.value_and_gradient!!` returns `(val, (∂f, ∂x))`; `∂f` is `NoTangent`
# because we registered `tangent_type(::Type{<:NamedTupleEvaluator}) = NoTangent`
# above, so the cache never carries a tangent for the user's problem.
# Shape validation is delegated to the inner `NamedTupleEvaluator{CheckInput}`
# callable Mooncake invokes — gated by the user's `check_dims` choice.
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
# raw shape, sidestepping the fixed `evaluator(x)` overhead.
@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:AutoMooncake,<:VectorEvaluator,<:MooncakeLoweredCache},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    c = p.cache
    val, tangents = Mooncake.value_and_gradient!!(
        c.cache, c.f, x, c.contexts...; args_to_zero=c.args_to_zero
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

# `raw_gradient_target` is a scalar-only fast path; jacobians must use the
# generic preparation.
@inline function AbstractPPL.value_and_jacobian!!(
    ::Prepared{<:AutoMooncake,<:VectorEvaluator,<:MooncakeLoweredCache},
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
