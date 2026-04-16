module AbstractPPLMooncakeExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes: AutoMooncake
using Mooncake: Mooncake

struct MooncakePrepared{E,C,P}
    evaluator::E
    cache::C
    inputspec::P
end

AbstractPPL.capabilities(::Type{<:MooncakePrepared}) = DerivativeOrder{1}()

function AbstractPPL.dimension(::MooncakePrepared{<:Any,<:Any,<:NamedTuple})
    throw(
        ArgumentError(
            "`dimension` is only available for evaluators prepared with a vector of floating-point numbers.",
        ),
    )
end
function AbstractPPL.dimension(p::MooncakePrepared{<:Any,<:Any,<:AbstractVector})
    return length(p.inputspec)
end

function (p::MooncakePrepared{<:Any,<:Any,<:NamedTuple})(values::NamedTuple)
    return p.evaluator(values)
end

function (p::MooncakePrepared{<:Any,<:Any,<:AbstractVector})(
    x::AbstractVector{<:AbstractFloat}
)
    length(x) == length(p.inputspec) || throw(
        DimensionMismatch(
            "Expected a vector of length $(length(p.inputspec)), but got length $(length(x)).",
        ),
    )
    return p.evaluator(x)
end

function AbstractPPL.prepare(adtype::AutoMooncake, problem, values::NamedTuple)
    evaluator = AbstractPPL.prepare(problem, values)
    cache = Mooncake.prepare_gradient_cache(evaluator, values; config=adtype.config)
    return MooncakePrepared(evaluator, cache, values)
end

function AbstractPPL.prepare(
    adtype::AutoMooncake, problem, x::AbstractVector{<:AbstractFloat}
)
    evaluator = AbstractPPL.prepare(problem, x)
    cache = Mooncake.prepare_gradient_cache(evaluator, x; config=adtype.config)
    return MooncakePrepared(evaluator, cache, x)
end

@inline function AbstractPPL.value_and_gradient(
    p::MooncakePrepared{<:Any,<:Any,<:NamedTuple}, values::NamedTuple
)
    typeof(values) === typeof(p.inputspec) || throw(
        ArgumentError(
            "Expected the same NamedTuple structure that was used to prepare this evaluator.",
        ),
    )
    val, (_, grad) = Mooncake.value_and_gradient!!(p.cache, p.evaluator, values)
    return (val, grad)
end

@inline function AbstractPPL.value_and_gradient(
    p::MooncakePrepared{<:Any,<:Any,<:AbstractVector}, x::AbstractVector{<:AbstractFloat}
)
    length(x) == length(p.inputspec) || throw(
        DimensionMismatch(
            "Expected a vector of length $(length(p.inputspec)), but got length $(length(x)).",
        ),
    )
    val, (_, grad) = Mooncake.value_and_gradient!!(p.cache, p.evaluator, x)
    return (val, grad)
end

end # module
