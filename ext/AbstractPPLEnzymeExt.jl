module AbstractPPLEnzymeExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes: AutoEnzyme
using Enzyme: Enzyme

struct EnzymePrepared{E}
    evaluator::E
end

AbstractPPL.capabilities(::Type{<:EnzymePrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::EnzymePrepared) = AbstractPPL.dimension(p.evaluator)

function (p::EnzymePrepared)(x)
    return p.evaluator(x)
end

function AbstractPPL.prepare(::AutoEnzyme, problem, x::AbstractVector{<:AbstractFloat})
    evaluator = AbstractPPL.ADProblems.VectorEvaluator(
        AbstractPPL.prepare(problem, x), length(x)
    )
    return EnzymePrepared(evaluator)
end

@inline function AbstractPPL.value_and_gradient(
    p::EnzymePrepared, x::AbstractVector{<:AbstractFloat}
)
    dx = zero(x)
    result = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
        Enzyme.Const(p.evaluator),
        Enzyme.Active,
        Enzyme.Duplicated(x, dx),
    )
    val = result[2]  # The primal value is returned in the second tuple entry.
    return (val, dx)
end

end # module
