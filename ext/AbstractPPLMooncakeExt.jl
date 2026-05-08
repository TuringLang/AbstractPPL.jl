module AbstractPPLMooncakeExt

using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators:
    Evaluators,
    Prepared,
    VectorEvaluator,
    NamedTupleEvaluator,
    _ad_output_arity,
    _assert_namedtuple_shape
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake

const _MooncakeAD = Union{AutoMooncake,AutoMooncakeForward}

# Tag a Mooncake cache with the prepared evaluator's output arity (`:scalar`
# or `:vector`) so `value_and_gradient!!` / `value_and_jacobian!!` can raise
# helpful arity-mismatch errors instead of failing inside Mooncake.
struct MooncakeCache{A,C}
    cache::C
end
MooncakeCache{A}(cache::C) where {A,C} = MooncakeCache{A,C}(cache)

_mooncake_config(adtype) = adtype.config === nothing ? Mooncake.Config() : adtype.config

# `value_and_gradient!!` accepts either a reverse-mode gradient cache
# (AutoMooncake) or a forward-mode derivative cache (AutoMooncakeForward).
function _mooncake_gradient_cache(::AutoMooncake, f, x; config)
    return Mooncake.prepare_gradient_cache(f, x; config=config)
end
function _mooncake_gradient_cache(::AutoMooncakeForward, f, x; config)
    return Mooncake.prepare_derivative_cache(f, x; config=config)
end

# `value_and_jacobian!!`: reverse mode wants a pullback cache, forward mode
# wants a derivative cache.
function _mooncake_jacobian_cache(::AutoMooncake, f, x; config)
    return Mooncake.prepare_pullback_cache(f, x; config=config)
end
function _mooncake_jacobian_cache(::AutoMooncakeForward, f, x; config)
    return Mooncake.prepare_derivative_cache(f, x; config=config)
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
    adtype::_MooncakeAD, problem, x::AbstractVector{<:Real}; check_dims::Bool=true
)
    evaluator = AbstractPPL.prepare(problem, x; check_dims)::VectorEvaluator
    arity = _ad_output_arity(evaluator(x))
    # Mooncake builds no tape for length-zero `x`; tag with `Nothing` so the
    # empty-input methods below shortcut without invoking Mooncake.
    length(x) == 0 && return Prepared(adtype, evaluator, MooncakeCache{arity}(nothing))
    config = _mooncake_config(adtype)
    cache = if arity === :scalar
        _mooncake_gradient_cache(adtype, evaluator, x; config)
    else
        _mooncake_jacobian_cache(adtype, evaluator, x; config)
    end
    return Prepared(adtype, evaluator, MooncakeCache{arity}(cache))
end

# `Mooncake.value_and_gradient!!` returns `(val, (∂f, ∂x))`; we discard the
# function tangent `∂f` and surface only `∂x` as the user-facing gradient.
@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:_MooncakeAD,<:NamedTupleEvaluator}, values::NamedTuple
)
    _assert_namedtuple_shape(p.evaluator, values)
    val, (_, grad) = Mooncake.value_and_gradient!!(p.cache, p.evaluator, values)
    return (val, grad)
end

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

@inline function AbstractPPL.value_and_gradient!!(
    ::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:vector}},
    ::AbstractVector{<:Real},
)
    throw(ArgumentError("`value_and_gradient!!` requires a scalar-valued function."))
end

@inline function AbstractPPL.value_and_jacobian!!(
    ::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:scalar}},
    ::AbstractVector{<:Real},
)
    throw(ArgumentError("`value_and_jacobian!!` requires a vector-valued function."))
end

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
