module AbstractPPLEnzymeExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes: AutoEnzyme
using Enzyme: Enzyme

struct EnzymePrepared{E,G,M,S}
    evaluator::E
    gradient::G
    mode::M
    shadow::S
end

AbstractPPL.capabilities(::Type{<:EnzymePrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::EnzymePrepared) = AbstractPPL.dimension(p.evaluator)

(p::EnzymePrepared)(x) = p.evaluator(x)

# Resolve the mode requested via `AutoEnzyme(; mode=...)` into one that also
# returns the primal value, so that `value_and_gradient` can return both.
# When `mode === nothing` we default to `set_runtime_activity(ReverseWithPrimal)`;
# when the caller supplies a mode we honor it as-is (only adding `WithPrimal`),
# leaving runtime-activity / strict-activity choices entirely to the caller.
_enzyme_mode(::Nothing) = Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal)
_enzyme_mode(mode) = Enzyme.WithPrimal(mode)

# Forward mode precomputes a one-hot basis shadow once so the hot path can call
# `Enzyme.autodiff(..., BatchDuplicated(...))` directly; this is concretely typed,
# whereas `Enzyme.gradient` returns a `NamedTuple` inferred as `Any` through the wrapper.
_enzyme_shadow(::Enzyme.ForwardMode, x) = Enzyme.onehot(x)
_enzyme_shadow(_, _) = nothing

# Reverse mode reuses a persistent buffer passed as the `Duplicated` shadow.
_enzyme_gradient(::Enzyme.ReverseMode, x) = similar(x)
_enzyme_gradient(_, _) = nothing

function AbstractPPL.prepare(
    adtype::AutoEnzyme, problem, x::AbstractVector{<:AbstractFloat}; check_dims::Bool=true
)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator{check_dims}(
        AbstractPPL.prepare(problem, x), length(x)
    )
    mode = _enzyme_mode(adtype.mode)
    return EnzymePrepared(
        evaluator, _enzyme_gradient(mode, x), mode, _enzyme_shadow(mode, x)
    )
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
    length(p.shadow) == length(x) || throw(
        DimensionMismatch(
            "Expected a vector of length $(length(p.shadow)), but got length $(length(x)).",
        ),
    )
    derivs, val = Enzyme.autodiff(
        p.mode, Enzyme.Const(p.evaluator), Enzyme.BatchDuplicated(x, p.shadow)
    )
    return (val, collect(values(derivs)))
end

end # module
