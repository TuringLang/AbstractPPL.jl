module AbstractPPLMooncakeExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake

struct MooncakePrepared{E,C}
    evaluator::E
    cache::C
end

AbstractPPL.capabilities(::Type{<:MooncakePrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::MooncakePrepared) = AbstractPPL.dimension(p.evaluator)

function (p::MooncakePrepared)(x)
    return p.evaluator(x)
end

function _prepare(adtype, problem, values::NamedTuple)
    evaluator = AbstractPPL.ADProblems.NamedTupleEvaluator(
        AbstractPPL.prepare(problem, values), values
    )
    cache = Mooncake.prepare_gradient_cache(evaluator, values; config=adtype.config)
    return MooncakePrepared(evaluator, cache)
end

function _prepare(adtype, problem, x::AbstractVector{<:AbstractFloat})
    evaluator = AbstractPPL.ADProblems.VectorEvaluator(
        AbstractPPL.prepare(problem, x), length(x)
    )
    cache = Mooncake.prepare_gradient_cache(evaluator, x; config=adtype.config)
    return MooncakePrepared(evaluator, cache)
end

function AbstractPPL.prepare(adtype::AutoMooncake, problem, values::NamedTuple)
    return _prepare(adtype, problem, values)
end

function AbstractPPL.prepare(adtype::AutoMooncakeForward, problem, values::NamedTuple)
    return _prepare(adtype, problem, values)
end

function AbstractPPL.prepare(
    adtype::AutoMooncake, problem, x::AbstractVector{<:AbstractFloat}
)
    return _prepare(adtype, problem, x)
end

function AbstractPPL.prepare(
    adtype::AutoMooncakeForward, problem, x::AbstractVector{<:AbstractFloat}
)
    return _prepare(adtype, problem, x)
end

@inline function AbstractPPL.value_and_gradient(
    p::MooncakePrepared{<:AbstractPPL.ADProblems.NamedTupleEvaluator}, values::NamedTuple
)
    typeof(values) === typeof(p.evaluator.inputspec) || throw(
        ArgumentError(
            "Expected the same NamedTuple structure that was used to prepare this evaluator.",
        ),
    )
    val, (_, grad) = Mooncake.value_and_gradient!!(p.cache, p.evaluator, values)
    return (val, grad)
end

@inline function AbstractPPL.value_and_gradient(
    p::MooncakePrepared{<:AbstractPPL.ADProblems.VectorEvaluator},
    x::AbstractVector{<:AbstractFloat},
)
    AbstractPPL.dimension(p.evaluator) == length(x) || throw(
        DimensionMismatch(
            "Expected a vector of length $(AbstractPPL.dimension(p.evaluator)), but got length $(length(x)).",
        ),
    )
    val, (_, grad) = Mooncake.value_and_gradient!!(p.cache, p.evaluator, x)
    return (val, grad)
end

end # module
