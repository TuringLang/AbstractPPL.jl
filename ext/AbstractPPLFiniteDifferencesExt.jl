module AbstractPPLFiniteDifferencesExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using AbstractPPL.Utils: flatten_to!!, unflatten_to!!
using ADTypes: AutoFiniteDifferences
using FiniteDifferences: FiniteDifferences

struct FDPrepared{F,M,P}
    evaluator
    f::F
    fdm::M
    inputspec::P
end

AbstractPPL.capabilities(::Type{<:FDPrepared}) = DerivativeOrder{1}()

function AbstractPPL.test_grad(f, x::AbstractVector{<:AbstractFloat})
    return FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), f, x)[1]
end

function AbstractPPL.dimension(::FDPrepared{<:Any,<:Any,<:NamedTuple})
    throw(
        ArgumentError(
            "`dimension` is only available for evaluators prepared with a vector of floating-point numbers.",
        ),
    )
end
function AbstractPPL.dimension(p::FDPrepared{<:Any,<:Any,<:AbstractVector})
    return length(p.inputspec)
end

function (p::FDPrepared{<:Any,<:Any,<:NamedTuple})(values::NamedTuple)
    typeof(values) === typeof(p.inputspec) || throw(
        ArgumentError(
            "Expected the same NamedTuple structure that was used to prepare this evaluator.",
        ),
    )
    return p.evaluator(values)
end

function (p::FDPrepared{<:Any,<:Any,<:NamedTuple})(x::AbstractVector)
    return p.f(x)
end

function (p::FDPrepared{<:Any,<:Any,<:AbstractVector})(x::AbstractVector{<:AbstractFloat})
    length(x) == length(p.inputspec) || throw(
        DimensionMismatch(
            "Expected a vector of length $(length(p.inputspec)), but got length $(length(x)).",
        ),
    )
    return p.evaluator(x)
end

function (p::FDPrepared)(x)
    throw(MethodError(p, (x,)))
end

function AbstractPPL.prepare(adtype::AutoFiniteDifferences, problem, values::NamedTuple)
    evaluator = AbstractPPL.prepare(problem, values)
    f = x -> evaluator(unflatten_to!!(values, x))
    return FDPrepared(evaluator, f, adtype.fdm, values)
end

function AbstractPPL.prepare(
    adtype::AutoFiniteDifferences, problem, x::AbstractVector{<:AbstractFloat}
)
    evaluator = AbstractPPL.prepare(problem, x)
    return FDPrepared(evaluator, evaluator, adtype.fdm, x)
end

function AbstractPPL.value_and_gradient(
    p::FDPrepared{<:Any,<:Any,<:NamedTuple}, values::NamedTuple
)
    typeof(values) === typeof(p.inputspec) || throw(
        ArgumentError(
            "Expected the same NamedTuple structure that was used to prepare this evaluator.",
        ),
    )
    x = flatten_to!!(nothing, values)
    val = p.evaluator(values)
    grad = FiniteDifferences.grad(p.fdm, p.f, x)[1]
    return (val, unflatten_to!!(p.inputspec, grad))
end

function AbstractPPL.value_and_gradient(
    p::FDPrepared{<:Any,<:Any,<:AbstractVector}, x::AbstractVector{<:AbstractFloat}
)
    length(x) == length(p.inputspec) || throw(
        DimensionMismatch(
            "Expected a vector of length $(length(p.inputspec)), but got length $(length(x)).",
        ),
    )
    val = p.evaluator(x)
    grad = FiniteDifferences.grad(p.fdm, p.f, x)[1]
    return (val, grad)
end

end # module
