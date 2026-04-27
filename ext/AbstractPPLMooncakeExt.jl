module AbstractPPLMooncakeExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems:
    _assert_jacobian_output,
    _assert_namedtuple_shape,
    _assert_supported_output,
    _is_scalar_output
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake

const MooncakeAD = Union{AutoMooncake,AutoMooncakeForward}

struct MooncakePrepared{E,GC,JC} <: AbstractPPL.ADProblems.AbstractPrepared
    evaluator::E
    gradient_cache::GC
    jacobian_cache::JC
end

function _mooncake_config(adtype)
    config = adtype.config
    return config === nothing ? Mooncake.Config() : config
end

# Mooncake's `value_and_jacobian!!` needs a `prepare_pullback_cache` (reverse,
# AutoMooncake) or `prepare_derivative_cache` (forward, AutoMooncakeForward).
function _mooncake_jacobian_cache(::AutoMooncake, evaluator, x; config)
    return Mooncake.prepare_pullback_cache(evaluator, x; config=config)
end
function _mooncake_jacobian_cache(::AutoMooncakeForward, evaluator, x; config)
    return Mooncake.prepare_derivative_cache(evaluator, x; config=config)
end

function AbstractPPL.prepare(
    adtype::MooncakeAD, problem, values::NamedTuple; check_dims::Bool=true
)
    evaluator = AbstractPPL.ADProblems.NamedTupleEvaluator{check_dims}(
        AbstractPPL.prepare(problem, values), values
    )
    config = _mooncake_config(adtype)
    cache = Mooncake.prepare_gradient_cache(evaluator, values; config)
    return MooncakePrepared(evaluator, cache, nothing)
end

function AbstractPPL.prepare(
    adtype::MooncakeAD, problem, x::AbstractVector{<:Real}; check_dims::Bool=true
)
    raw = AbstractPPL.prepare(problem, x)
    length(x) == 0 && return AbstractPPL.ADProblems.VectorEvaluator{check_dims}(raw, 0)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator{check_dims}(raw, length(x))
    y = evaluator(x)
    _assert_supported_output(y)
    config = _mooncake_config(adtype)
    if _is_scalar_output(y)
        cache = Mooncake.prepare_gradient_cache(evaluator, x; config)
        return MooncakePrepared(evaluator, cache, nothing)
    else
        _assert_jacobian_output(y)
        cache = _mooncake_jacobian_cache(adtype, evaluator, x; config)
        return MooncakePrepared(evaluator, nothing, cache)
    end
end

@inline function AbstractPPL.value_and_gradient(
    p::MooncakePrepared{<:AbstractPPL.ADProblems.NamedTupleEvaluator}, values::NamedTuple
)
    _assert_namedtuple_shape(p.evaluator, values)
    p.gradient_cache === nothing &&
        throw(ArgumentError("`value_and_gradient` requires a scalar-valued function."))
    # `value_and_gradient!!` returns (val, (∂f, ∂x)); discard the function tangent ∂f.
    val, (_, grad) = Mooncake.value_and_gradient!!(p.gradient_cache, p.evaluator, values)
    return (val, grad)
end

@inline function AbstractPPL.value_and_gradient(
    p::MooncakePrepared{<:AbstractPPL.ADProblems.VectorEvaluator}, x::AbstractVector{<:Real}
)
    p.gradient_cache === nothing &&
        throw(ArgumentError("`value_and_gradient` requires a scalar-valued function."))
    # `value_and_gradient!!` returns (val, (∂f, ∂x)); discard the function tangent ∂f.
    val, (_, grad) = Mooncake.value_and_gradient!!(p.gradient_cache, p.evaluator, x)
    return (val, grad)
end

@inline function AbstractPPL.value_and_jacobian(
    p::MooncakePrepared{<:AbstractPPL.ADProblems.VectorEvaluator}, x::AbstractVector{<:Real}
)
    p.jacobian_cache === nothing &&
        throw(ArgumentError("`value_and_jacobian` requires a vector-valued function."))
    return Mooncake.value_and_jacobian!!(p.jacobian_cache, p.evaluator, x)
end

function AbstractPPL.ADProblems._supports_gradient(
    ::MooncakePrepared{<:AbstractPPL.ADProblems.VectorEvaluator,<:Any,Nothing}
)
    return true
end

end # module
