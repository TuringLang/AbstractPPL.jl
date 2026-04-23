module AbstractPPLMooncakeExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems: _assert_namedtuple_shape, _check_mode, _check_namedtuple_mode
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake

const MooncakeAD = Union{AutoMooncake,AutoMooncakeForward}

struct MooncakePrepared{Mode,E,C} <: AbstractPPL.ADProblems.AbstractPrepared{Mode}
    evaluator::E
    cache::C
    function MooncakePrepared{Mode}(evaluator::E, cache::C) where {Mode,E,C}
        return new{Mode,E,C}(evaluator, cache)
    end
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
    adtype::MooncakeAD,
    problem,
    values::NamedTuple;
    check_dims::Bool=true,
    mode::Symbol=:gradient,
)
    _check_namedtuple_mode(mode)
    evaluator = AbstractPPL.ADProblems.NamedTupleEvaluator{check_dims}(
        AbstractPPL.prepare(problem, values), values
    )
    cache = Mooncake.prepare_gradient_cache(
        evaluator, values; config=_mooncake_config(adtype)
    )
    return MooncakePrepared{:gradient}(evaluator, cache)
end

function AbstractPPL.prepare(
    adtype::MooncakeAD,
    problem,
    x::AbstractVector{<:AbstractFloat};
    check_dims::Bool=true,
    mode::Symbol=:gradient,
)
    _check_mode(mode)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator{check_dims}(
        AbstractPPL.prepare(problem, x), length(x)
    )
    if mode === :gradient
        cache = Mooncake.prepare_gradient_cache(
            evaluator, x; config=_mooncake_config(adtype)
        )
        return MooncakePrepared{:gradient}(evaluator, cache)
    else
        cache = _mooncake_jacobian_cache(
            adtype, evaluator, x; config=_mooncake_config(adtype)
        )
        return MooncakePrepared{:jacobian}(evaluator, cache)
    end
end

@inline function AbstractPPL.value_and_gradient(
    p::MooncakePrepared{:gradient,<:AbstractPPL.ADProblems.NamedTupleEvaluator},
    values::NamedTuple,
)
    _assert_namedtuple_shape(p.evaluator, values)
    val, (_, grad) = Mooncake.value_and_gradient!!(p.cache, p.evaluator, values)
    return (val, grad)
end

@inline function AbstractPPL.value_and_gradient(
    p::MooncakePrepared{:gradient,<:AbstractPPL.ADProblems.VectorEvaluator},
    x::AbstractVector{<:AbstractFloat},
)
    val, (_, grad) = Mooncake.value_and_gradient!!(p.cache, p.evaluator, x)
    return (val, grad)
end

@inline function AbstractPPL.value_and_jacobian(
    p::MooncakePrepared{:jacobian,<:AbstractPPL.ADProblems.VectorEvaluator},
    x::AbstractVector{<:AbstractFloat},
)
    return Mooncake.value_and_jacobian!!(p.cache, p.evaluator, x)
end

end # module
