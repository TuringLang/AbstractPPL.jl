module AbstractPPLEnzymeExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems: _check_mode
using ADTypes: AutoEnzyme
using Enzyme: Enzyme

struct EnzymePrepared{Mode,E,G,M,S} <: AbstractPPL.ADProblems.AbstractPrepared{Mode}
    evaluator::E
    gradient::G
    mode::M
    shadow::S
    function EnzymePrepared{Mode}(
        evaluator::E, gradient::G, mode::M, shadow::S
    ) where {Mode,E,G,M,S}
        return new{Mode,E,G,M,S}(evaluator, gradient, mode, shadow)
    end
end

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

# Reverse mode pre-allocates the gradient buffer to avoid per-call allocation on the hot path.
_enzyme_gradient(::Enzyme.ReverseMode, x) = similar(x)
_enzyme_gradient(_, _) = nothing

function AbstractPPL.prepare(
    adtype::AutoEnzyme,
    problem,
    x::AbstractVector{<:AbstractFloat};
    check_dims::Bool=true,
    mode::Symbol=:gradient,
)
    _check_mode(mode)
    length(x) == 0 && return AbstractPPL.ADProblems.VectorEvaluator{check_dims}(
        AbstractPPL.prepare(problem, x), 0
    )
    evaluator = AbstractPPL.ADProblems.VectorEvaluator{check_dims}(
        AbstractPPL.prepare(problem, x), length(x)
    )
    enzyme_mode = _enzyme_mode(adtype.mode)
    if mode === :gradient
        return EnzymePrepared{:gradient}(
            evaluator,
            _enzyme_gradient(enzyme_mode, x),
            enzyme_mode,
            _enzyme_shadow(enzyme_mode, x),
        )
    else
        # Jacobian uses `Enzyme.jacobian` sugar; no buffer or shadow to keep.
        return EnzymePrepared{:jacobian}(evaluator, nothing, enzyme_mode, nothing)
    end
end

@inline function AbstractPPL.value_and_gradient(
    p::EnzymePrepared{:gradient,<:Any,<:Any,<:Enzyme.ReverseMode},
    x::AbstractVector{<:AbstractFloat},
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
    p::EnzymePrepared{:gradient,<:Any,<:Any,<:Enzyme.ForwardMode},
    x::AbstractVector{<:AbstractFloat},
)
    length(p.shadow) == length(x) || throw(
        DimensionMismatch(
            "Expected a vector of length $(length(p.shadow)), but got length $(length(x)).",
        ),
    )
    derivs, val = Enzyme.autodiff(
        p.mode, Enzyme.Const(p.evaluator), Enzyme.BatchDuplicated(x, p.shadow)
    )
    # length-1 input yields a scalar; wrap it so the gradient is always a Vector.
    grad = derivs isa Number ? [derivs] : collect(values(derivs))
    return (val, grad)
end

@inline function AbstractPPL.value_and_jacobian(
    p::EnzymePrepared{:jacobian}, x::AbstractVector{<:AbstractFloat}
)
    # `Enzyme.jacobian(WithPrimal(mode), f, x)` returns `(derivs=(J,), val=y)`
    # in both forward and reverse modes.
    nt = Enzyme.jacobian(p.mode, Enzyme.Const(p.evaluator), x)
    return (nt.val, nt.derivs[1])
end

end # module
