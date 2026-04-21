module AbstractPPLFiniteDifferencesExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using AbstractPPL.ADProblems: _assert_namedtuple_shape
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
    _assert_namedtuple_shape(p.evaluator, values)
    return p.evaluator(values)
end

(p::FDPrepared)(x) = p.evaluator(x)

function AbstractPPL.prepare(
    adtype::AutoFiniteDifferences, problem, values::NamedTuple; check_dims::Bool=true
)
    evaluator = AbstractPPL.ADProblems.NamedTupleEvaluator{check_dims}(
        AbstractPPL.prepare(problem, values), values
    )
    f = x -> evaluator(unflatten_to!!(values, x))
    return FDPrepared(evaluator, f, adtype.fdm)
end

function AbstractPPL.prepare(
    adtype::AutoFiniteDifferences,
    problem,
    x::AbstractVector{<:AbstractFloat};
    check_dims::Bool=true,
)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator{check_dims}(
        AbstractPPL.prepare(problem, x), length(x)
    )
    return FDPrepared(evaluator, evaluator, adtype.fdm)
end

function AbstractPPL.value_and_gradient(
    p::FDPrepared{<:AbstractPPL.ADProblems.NamedTupleEvaluator}, values::NamedTuple
)
    _assert_namedtuple_shape(p.evaluator, values)
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
