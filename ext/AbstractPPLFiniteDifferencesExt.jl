module AbstractPPLFiniteDifferencesExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using AbstractPPL.Utils: flatten_to!!, unflatten_to!!
using ADTypes: AutoFiniteDifferences
using FiniteDifferences: FiniteDifferences

struct FDPrepared{E,F,M}
    evaluator::E
    f::F
    fdm::M
end

AbstractPPL.capabilities(::Type{<:FDPrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::FDPrepared) = AbstractPPL.dimension(p.evaluator)

function (p::FDPrepared{<:AbstractPPL.ADProblems.NamedTupleEvaluator})(values::NamedTuple)
    typeof(values) === typeof(p.evaluator.inputspec) || throw(
        ArgumentError(
            "Expected the same NamedTuple structure that was used to prepare this evaluator.",
        ),
    )
    return p.evaluator(values)
end

function (p::FDPrepared)(x)
    return p.evaluator(x)
end

function AbstractPPL.prepare(adtype::AutoFiniteDifferences, problem, values::NamedTuple)
    evaluator = AbstractPPL.ADProblems.NamedTupleEvaluator(
        AbstractPPL.prepare(problem, values), values
    )
    f = x -> evaluator(unflatten_to!!(values, x))
    return FDPrepared(evaluator, f, adtype.fdm)
end

function AbstractPPL.prepare(
    adtype::AutoFiniteDifferences, problem, x::AbstractVector{<:AbstractFloat}
)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator(
        AbstractPPL.prepare(problem, x), length(x)
    )
    return FDPrepared(evaluator, evaluator, adtype.fdm)
end

function AbstractPPL.value_and_gradient(
    p::FDPrepared{<:AbstractPPL.ADProblems.NamedTupleEvaluator}, values::NamedTuple
)
    typeof(values) === typeof(p.evaluator.inputspec) || throw(
        ArgumentError(
            "Expected the same NamedTuple structure that was used to prepare this evaluator.",
        ),
    )
    x = flatten_to!!(nothing, values)
    val = p.evaluator(values)
    grad = FiniteDifferences.grad(p.fdm, p.f, x)[1]
    return (val, unflatten_to!!(p.evaluator.inputspec, grad))
end

function AbstractPPL.value_and_gradient(
    p::FDPrepared{<:AbstractPPL.ADProblems.VectorEvaluator},
    x::AbstractVector{<:AbstractFloat},
)
    val = p.evaluator(x)
    grad = FiniteDifferences.grad(p.fdm, p.f, x)[1]
    return (val, grad)
end

end # module
