module AbstractPPLEnzymeExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes: AutoEnzyme
using Enzyme: Enzyme

struct EnzymePrepared{E,G,M}
    evaluator::E
    gradient::G
    mode::M
end

AbstractPPL.capabilities(::Type{<:EnzymePrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::EnzymePrepared) = AbstractPPL.dimension(p.evaluator)

(p::EnzymePrepared)(x) = p.evaluator(x)

# Resolve the mode requested via `AutoEnzyme(; mode=...)` into one that also
# returns the primal value, so that `value_and_gradient` can return both.
_enzyme_mode(::Nothing) = Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal)
_enzyme_mode(mode) = Enzyme.WithPrimal(mode)

function AbstractPPL.prepare(
    adtype::AutoEnzyme, problem, x::AbstractVector{<:AbstractFloat}
)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator(
        AbstractPPL.prepare(problem, x), length(x)
    )
    return EnzymePrepared(evaluator, similar(x), _enzyme_mode(adtype.mode))
end

@inline function AbstractPPL.value_and_gradient(
    p::EnzymePrepared{<:Any,<:Any,<:Enzyme.ReverseMode}, x::AbstractVector{<:AbstractFloat}
)
    dx = p.gradient
    length(dx) == length(x) || throw(
        DimensionMismatch(
            "Expected a vector of length $(length(dx)), but got length $(length(x))."
        ),
    )
    fill!(dx, zero(eltype(dx)))
    result = Enzyme.autodiff(
        p.mode, Enzyme.Const(p.evaluator), Enzyme.Active, Enzyme.Duplicated(x, dx)
    )
    return (result[2], copy(dx))
end

@inline function AbstractPPL.value_and_gradient(
    p::EnzymePrepared{<:Any,<:Any,<:Enzyme.ForwardMode}, x::AbstractVector{<:AbstractFloat}
)
    out = Enzyme.gradient(p.mode, Enzyme.Const(p.evaluator), x)
    return (out.val, only(out.derivs))
end

end # module
