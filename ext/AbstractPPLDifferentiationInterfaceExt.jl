module AbstractPPLDifferentiationInterfaceExt

using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators: Evaluators, Prepared, VectorEvaluator, _ad_output_arity
using ADTypes: AbstractADType, AutoReverseDiff
using DifferentiationInterface: DifferentiationInterface as DI

# Differentiate only `x`; the evaluator is passed as a `DI.Constant` context so
# that in DynamicPPL the model and other evaluator state stay constant.
@inline _call_evaluator(x, evaluator) = evaluator(x)

struct DICache{F,GP,JP}
    target::F
    gradient_prep::GP
    jacobian_prep::JP
    use_context::Bool
end

# Compiled ReverseDiff only reuses a compiled tape on the one-argument path;
# `DI.Constant` deactivates tape recording, so close the evaluator into the
# target and call DI without contexts.
function _prepare_di(prep::F, adtype::AutoReverseDiff{true}, x, evaluator) where {F}
    target = Base.Fix2(_call_evaluator, evaluator)
    return target, prep(target, adtype, x), false
end

function _prepare_di(prep::F, adtype::AbstractADType, x, evaluator) where {F}
    return _call_evaluator, prep(_call_evaluator, adtype, x, DI.Constant(evaluator)), true
end

function AbstractPPL.prepare(
    adtype::AbstractADType, problem, x::AbstractVector{<:Real}; check_dims::Bool=true
)
    evaluator = AbstractPPL.prepare(problem, x; check_dims)::VectorEvaluator
    arity = _ad_output_arity(evaluator(x))
    if length(x) == 0
        # DI prep crashes on length-0 input (e.g. ForwardDiff `BoundsError`); the
        # `Val(0)` sentinel keeps the `gradient_prep === nothing` arity check meaningful.
        gp, jp = arity === :scalar ? (Val(0), nothing) : (nothing, Val(0))
        return Prepared(adtype, evaluator, DICache(_call_evaluator, gp, jp, true))
    end
    if arity === :scalar
        target, gradient_prep, use_context = _prepare_di(
            DI.prepare_gradient, adtype, x, evaluator
        )
        return Prepared(
            adtype, evaluator, DICache(target, gradient_prep, nothing, use_context)
        )
    end
    target, jacobian_prep, use_context = _prepare_di(
        DI.prepare_jacobian, adtype, x, evaluator
    )
    return Prepared(adtype, evaluator, DICache(target, nothing, jacobian_prep, use_context))
end

@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:AbstractADType,<:VectorEvaluator,<:DICache}, x::AbstractVector{T}
) where {T<:Real}
    p.cache.gradient_prep === nothing && Evaluators._throw_gradient_needs_scalar()
    Evaluators._check_ad_input(p.evaluator, x)
    # Bypass DI on length-0 input — DI prep paths fail (e.g. ForwardDiff
    # `BoundsError`); typed `T[]` matches the caller's element type.
    length(x) == 0 && return (p.evaluator(x), T[])
    return if p.cache.use_context
        DI.value_and_gradient(
            p.cache.target, p.cache.gradient_prep, p.adtype, x, DI.Constant(p.evaluator)
        )
    else
        DI.value_and_gradient(p.cache.target, p.cache.gradient_prep, p.adtype, x)
    end
end

@inline function AbstractPPL.value_and_jacobian!!(
    p::Prepared{<:AbstractADType,<:VectorEvaluator,<:DICache}, x::AbstractVector{T}
) where {T<:Real}
    p.cache.jacobian_prep === nothing && Evaluators._throw_jacobian_needs_vector()
    Evaluators._check_ad_input(p.evaluator, x)
    if length(x) == 0
        val = p.evaluator(x)
        return (val, similar(x, length(val), 0))
    end
    return if p.cache.use_context
        DI.value_and_jacobian(
            p.cache.target, p.cache.jacobian_prep, p.adtype, x, DI.Constant(p.evaluator)
        )
    else
        DI.value_and_jacobian(p.cache.target, p.cache.jacobian_prep, p.adtype, x)
    end
end

end # module
