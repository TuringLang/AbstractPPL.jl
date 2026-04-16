module AbstractPPLEnzymeExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes: AutoEnzyme
using Enzyme: Enzyme

struct EnzymePrepared{E}
    evaluator::E
    dim::Int
end

AbstractPPL.capabilities(::Type{<:EnzymePrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::EnzymePrepared) = p.dim

function (p::EnzymePrepared)(x::AbstractVector{<:AbstractFloat})
    length(x) == p.dim || throw(
        DimensionMismatch(
            "Expected a vector of length $(p.dim), but got length $(length(x))."
        ),
    )
    return p.evaluator(x)
end

function AbstractPPL.prepare(::AutoEnzyme, problem, x::AbstractVector{<:AbstractFloat})
    evaluator = AbstractPPL.prepare(problem, x)
    return EnzymePrepared(evaluator, length(x))
end

@inline function AbstractPPL.value_and_gradient(
    p::EnzymePrepared, x::AbstractVector{<:AbstractFloat}
)
    length(x) == p.dim || throw(
        DimensionMismatch(
            "Expected a vector of length $(p.dim), but got length $(length(x))."
        ),
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
