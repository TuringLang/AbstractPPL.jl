module AbstractPPLEnzymeExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems: _assert_supported_output, _is_vector_output
using ADTypes: AutoEnzyme
using Enzyme: Enzyme

struct EnzymePrepared{E,G,M,S,IsVectorValued} <: AbstractPPL.ADProblems.AbstractPrepared
    evaluator::E
    gradient::G
    mode::M
    shadow::S
    function EnzymePrepared(
        evaluator::E, gradient::G, mode::M, shadow::S, ::Val{IsVectorValued}
    ) where {E,G,M,S,IsVectorValued}
        return new{E,G,M,S,IsVectorValued}(evaluator, gradient, mode, shadow)
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
    adtype::AutoEnzyme, problem, x::AbstractVector{<:Real}; check_dims::Bool=true
)
    raw = AbstractPPL.prepare(problem, x)
    length(x) == 0 && return AbstractPPL.ADProblems.VectorEvaluator{check_dims}(raw, 0)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator{check_dims}(raw, length(x))
    enzyme_mode = _enzyme_mode(adtype.mode)
    y = evaluator(x)
    _assert_supported_output(y)
    if _is_vector_output(y)
        return EnzymePrepared(evaluator, nothing, enzyme_mode, nothing, Val(true))
    else
        return EnzymePrepared(
            evaluator,
            _enzyme_gradient(enzyme_mode, x),
            enzyme_mode,
            _enzyme_shadow(enzyme_mode, x),
            Val(false),
        )
    end
end

@inline function AbstractPPL.value_and_gradient(
    p::EnzymePrepared{<:Any,<:Any,<:Enzyme.ReverseMode,<:Any,false},
    x::AbstractVector{<:Real},
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
    # ReverseWithPrimal returns (pullback_result, primal); index 2 is the primal.
    return (result[2], copy(dx))
end

@inline function AbstractPPL.value_and_gradient(
    p::EnzymePrepared{<:Any,<:Any,<:Enzyme.ForwardMode,<:Any,false},
    x::AbstractVector{<:Real},
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

function AbstractPPL.value_and_gradient(
    ::EnzymePrepared{<:Any,<:Any,<:Any,<:Any,true}, ::AbstractVector{<:Real}
)
    throw(ArgumentError("`value_and_gradient` requires a scalar-valued function."))
end

@inline function AbstractPPL.value_and_jacobian(
    p::EnzymePrepared{<:Any,<:Any,<:Any,<:Any,true}, x::AbstractVector{<:Real}
)
    # `Enzyme.jacobian(WithPrimal(mode), f, x)` returns `(derivs=(J,), val=y)`
    # in both forward and reverse modes.
    nt = Enzyme.jacobian(p.mode, Enzyme.Const(p.evaluator), x)
    return (nt.val, nt.derivs[1])
end

function AbstractPPL.value_and_jacobian(
    ::EnzymePrepared{<:Any,<:Any,<:Any,<:Any,false}, ::AbstractVector{<:Real}
)
    throw(ArgumentError("`value_and_jacobian` requires a vector-valued function."))
end

function AbstractPPL.ADProblems._supports_gradient(
    ::EnzymePrepared{<:Any,<:Any,<:Any,<:Any,false}
)
    return true
end

end # module
