module AbstractPPLMooncakeExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using AbstractPPL.ADProblems: _assert_namedtuple_shape
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake

const MooncakeAD = Union{AutoMooncake,AutoMooncakeForward}

struct MooncakePrepared{E,C}
    evaluator::E
    cache::C
end

AbstractPPL.capabilities(::Type{<:MooncakePrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::MooncakePrepared) = AbstractPPL.dimension(p.evaluator)

function (p::MooncakePrepared{<:AbstractPPL.ADProblems.NamedTupleEvaluator})(
    values::NamedTuple
)
    _assert_namedtuple_shape(p.evaluator, values)
    return p.evaluator(values)
end

(p::MooncakePrepared)(x) = p.evaluator(x)

function _mooncake_config(adtype)
    config = adtype.config
    return config === nothing ? Mooncake.Config() : config
end

function AbstractPPL.prepare(
    adtype::MooncakeAD, problem, values::NamedTuple; check_dims::Bool=true
)
    evaluator = AbstractPPL.ADProblems.NamedTupleEvaluator{check_dims}(
        AbstractPPL.prepare(problem, values), values
    )
    cache = Mooncake.prepare_gradient_cache(
        evaluator, values; config=_mooncake_config(adtype)
    )
    return MooncakePrepared(evaluator, cache)
end

function AbstractPPL.prepare(
    adtype::MooncakeAD, problem, x::AbstractVector{<:AbstractFloat}; check_dims::Bool=true
)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator{check_dims}(
        AbstractPPL.prepare(problem, x), length(x)
    )
    cache = Mooncake.prepare_gradient_cache(evaluator, x; config=_mooncake_config(adtype))
    return MooncakePrepared(evaluator, cache)
end

@inline function AbstractPPL.value_and_gradient(
    p::MooncakePrepared{<:AbstractPPL.ADProblems.NamedTupleEvaluator}, values::NamedTuple
)
    _assert_namedtuple_shape(p.evaluator, values)
    val, (_, grad) = Mooncake.value_and_gradient!!(p.cache, p.evaluator, values)
    return (val, grad)
end

@inline function AbstractPPL.value_and_gradient(
    p::MooncakePrepared{<:AbstractPPL.ADProblems.VectorEvaluator},
    x::AbstractVector{<:AbstractFloat},
)
    val, (_, grad) = Mooncake.value_and_gradient!!(p.cache, p.evaluator, x)
    return (val, grad)
end

end # module
