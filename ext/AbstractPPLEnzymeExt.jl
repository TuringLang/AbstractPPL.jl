module AbstractPPLEnzymeExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes: AutoEnzyme
using Enzyme: Enzyme

struct EnzymePrepared{E,G}
    evaluator::E
    gradient::G
end

AbstractPPL.capabilities(::Type{<:EnzymePrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::EnzymePrepared) = AbstractPPL.dimension(p.evaluator)

(p::EnzymePrepared)(x) = p.evaluator(x)

function AbstractPPL.prepare(::AutoEnzyme, problem, x::AbstractVector{<:AbstractFloat})
    evaluator = AbstractPPL.ADProblems.VectorEvaluator(
        AbstractPPL.prepare(problem, x), length(x)
    )
    return EnzymePrepared(evaluator, similar(x))
end

@inline function AbstractPPL.value_and_gradient(
    p::EnzymePrepared, x::AbstractVector{<:AbstractFloat}
)
    dx = p.gradient
    length(dx) == length(x) || throw(
        DimensionMismatch(
            "Expected a vector of length $(length(dx)), but got length $(length(x))."
        ),
    )
    fill!(dx, zero(eltype(dx)))
    result = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
        Enzyme.Const(p.evaluator),
        Enzyme.Active,
        Enzyme.Duplicated(x, dx),
    )
    return (result[2], copy(dx))
end

end # module
