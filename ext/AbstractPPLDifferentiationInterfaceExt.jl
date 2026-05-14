module AbstractPPLDifferentiationInterfaceExt

using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators: Evaluators, Prepared, VectorEvaluator, _ad_output_arity
using ADTypes: AbstractADType, AutoReverseDiff
using DifferentiationInterface: DifferentiationInterface as DI

# Differentiate only `x`; the evaluator is passed as a `DI.Constant` context so
# that in DynamicPPL the model and other evaluator state stay constant.
@inline _call_evaluator(x, evaluator) = evaluator(x)

# `UseContext` is type-encoded so the dispatch between the context and
# no-context DI call is resolved at compile time; on tiny problems the runtime
# branch would otherwise show up as fixed overhead in the AD hot path.
struct DICache{UseContext,F,GP,JP}
    target::F
    gradient_prep::GP
    jacobian_prep::JP
    function DICache{UseContext}(target::F, gp::GP, jp::JP) where {UseContext,F,GP,JP}
        return new{UseContext,F,GP,JP}(target, gp, jp)
    end
end

# Compiled ReverseDiff only reuses a compiled tape on the one-argument path;
# `DI.Constant` deactivates tape recording, so close the evaluator into the
# target and call DI without contexts. The trailing `Val(false)`/`Val(true)`
# carries `UseContext` to the `DICache` constructor at compile time.
function _prepare_di(prep::F, adtype::AutoReverseDiff{true}, x, evaluator) where {F}
    target = Base.Fix2(_call_evaluator, evaluator)
    return target, prep(target, adtype, x), Val(false)
end

function _prepare_di(prep::F, adtype::AbstractADType, x, evaluator) where {F}
    return (
        _call_evaluator, prep(_call_evaluator, adtype, x, DI.Constant(evaluator)), Val(true)
    )
end

@inline _wrap_cache(target, gp, jp, ::Val{UseContext}) where {UseContext} =
    DICache{UseContext}(target, gp, jp)

# `raw_gradient_target` is accepted for signature parity with the Mooncake
# extension's vector `prepare`, but DI has no equivalent context-lowering
# entry — only `nothing` is supported here.
function AbstractPPL.prepare(
    adtype::AbstractADType,
    problem,
    x::AbstractVector{<:Real};
    check_dims::Bool=true,
    raw_gradient_target=nothing,
)
    raw_gradient_target === nothing || throw(
        ArgumentError(
            "`raw_gradient_target` is not supported by the DifferentiationInterface extension.",
        ),
    )
    evaluator = AbstractPPL.prepare(problem, x; check_dims)::VectorEvaluator
    arity = _ad_output_arity(evaluator(x))
    if length(x) == 0
        # DI prep crashes on length-0 input (e.g. ForwardDiff `BoundsError`); the
        # `Val(0)` sentinel keeps the `gradient_prep === nothing` arity check
        # meaningful. `UseContext` is irrelevant on this shortcut path — the AD
        # entry returns `(p.evaluator(x), T[])` before any DI call.
        gp, jp = arity === :scalar ? (Val(0), nothing) : (nothing, Val(0))
        return Prepared(adtype, evaluator, DICache{true}(_call_evaluator, gp, jp))
    end
    if arity === :scalar
        target, gradient_prep, ctx = _prepare_di(DI.prepare_gradient, adtype, x, evaluator)
        return Prepared(adtype, evaluator, _wrap_cache(target, gradient_prep, nothing, ctx))
    end
    target, jacobian_prep, ctx = _prepare_di(DI.prepare_jacobian, adtype, x, evaluator)
    return Prepared(adtype, evaluator, _wrap_cache(target, nothing, jacobian_prep, ctx))
end

# Compile-time dispatch on the `UseContext` type parameter eliminates the
# context-vs-no-context branch from the AD hot path.
@inline _di_value_and_gradient(c::DICache{true}, ad, x, eval) =
    DI.value_and_gradient(c.target, c.gradient_prep, ad, x, DI.Constant(eval))
@inline _di_value_and_gradient(c::DICache{false}, ad, x, _) =
    DI.value_and_gradient(c.target, c.gradient_prep, ad, x)

@inline _di_value_and_jacobian(c::DICache{true}, ad, x, eval) =
    DI.value_and_jacobian(c.target, c.jacobian_prep, ad, x, DI.Constant(eval))
@inline _di_value_and_jacobian(c::DICache{false}, ad, x, _) =
    DI.value_and_jacobian(c.target, c.jacobian_prep, ad, x)

@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:AbstractADType,<:VectorEvaluator,<:DICache}, x::AbstractVector{T}
) where {T<:Real}
    p.cache.gradient_prep === nothing && Evaluators._throw_gradient_needs_scalar()
    Evaluators._check_ad_input(p.evaluator, x)
    # Bypass DI on length-0 input — DI prep paths fail (e.g. ForwardDiff
    # `BoundsError`); typed `T[]` matches the caller's element type.
    length(x) == 0 && return (p.evaluator(x), T[])
    return _di_value_and_gradient(p.cache, p.adtype, x, p.evaluator)
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
    return _di_value_and_jacobian(p.cache, p.adtype, x, p.evaluator)
end

end # module
