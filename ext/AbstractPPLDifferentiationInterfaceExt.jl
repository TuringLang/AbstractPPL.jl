module AbstractPPLDifferentiationInterfaceExt

using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators: Evaluators, Prepared, VectorEvaluator, _ad_output_arity
using ADTypes: AbstractADType, AutoReverseDiff
using DifferentiationInterface: DifferentiationInterface as DI

# AD target used by both `DICache` modes. `Vararg{Any,N}` with a free `N`
# forces specialization on the trailing arity (a bare `Vararg{Any}` would
# skip it). DI invokes this as `_call_evaluator(x, f, c1, …, cN)` on the
# constants path, and as `_call_evaluator(x, evaluator)` (via `Fix2`) on
# the closure path — empty `ctx` then makes the splat a no-op.
@inline _call_evaluator(x, f::F, ctx::Vararg{Any,N}) where {F,N} = f(x, ctx...)

# `Mode` tags the cache shape:
#   * `:closure`    — compiled-tape ReverseDiff: target is a `Fix2` closure,
#                     the AD call passes **0** `DI.Constant`s.
#   * `N::Int`      — constants path: `N == length(evaluator.context)`, the
#                     AD call passes **N + 1** `DI.Constant`s (`f` plus the
#                     `N` context values).
# Encoding `Mode` in the type resolves the dispatch in `_di_value_and_*`
# at compile time without a runtime branch.
struct DICache{Mode,F,GP,JP}
    target::F
    gradient_prep::GP
    jacobian_prep::JP
    function DICache{Mode}(target::F, gp::GP, jp::JP) where {Mode,F,GP,JP}
        return new{Mode,F,GP,JP}(target, gp, jp)
    end
end

# Compiled ReverseDiff only reuses a compiled tape on the one-argument path;
# `DI.Constant` deactivates tape recording, so close the evaluator into the
# target and call DI without constants. Context (if any) is captured inside
# the evaluator closure rather than lowered out — the lowered path would also
# require a closure here, so the wrapper cost is unavoidable for compiled tapes.
function _prepare_di(prep::F, adtype::AutoReverseDiff{true}, x, evaluator) where {F}
    target = Base.Fix2(_call_evaluator, evaluator)
    return target, prep(target, adtype, x), Val(:closure)
end

function _prepare_di(prep::F, adtype::AbstractADType, x, evaluator) where {F}
    constants = (DI.Constant(evaluator.f), map(DI.Constant, evaluator.context)...)
    return (
        _call_evaluator,
        prep(_call_evaluator, adtype, x, constants...),
        Val(length(evaluator.context)),
    )
end

@inline _wrap_cache(target, gp, jp, ::Val{Mode}) where {Mode} =
    DICache{Mode}(target, gp, jp)

function AbstractPPL.prepare(
    adtype::AbstractADType,
    problem,
    x::AbstractVector{<:Real};
    check_dims::Bool=true,
    context::Tuple=(),
)
    evaluator = AbstractPPL.prepare(problem, x; check_dims, context)::VectorEvaluator
    arity = _ad_output_arity(evaluator(x))
    if length(x) == 0
        # DI prep crashes on length-0 input (e.g. ForwardDiff `BoundsError`).
        # `Val(0)` is an arity sentinel for the `gradient_prep === nothing`
        # check below; the AD entry short-circuits before any DI call.
        gp, jp = arity === :scalar ? (Val(0), nothing) : (nothing, Val(0))
        cache = _wrap_cache(_call_evaluator, gp, jp, Val(length(context)))
        return Prepared(adtype, evaluator, cache)
    end
    if arity === :scalar
        target, gradient_prep, mode = _prepare_di(DI.prepare_gradient, adtype, x, evaluator)
        return Prepared(
            adtype, evaluator, _wrap_cache(target, gradient_prep, nothing, mode)
        )
    end
    target, jacobian_prep, mode = _prepare_di(DI.prepare_jacobian, adtype, x, evaluator)
    return Prepared(adtype, evaluator, _wrap_cache(target, nothing, jacobian_prep, mode))
end

# Hot-path dispatch is by `Mode` (closure vs constants), resolved at compile
# time. The unconstrained method matches every non-`:closure` `Mode` (i.e.
# any `Int N`); `:closure` is strictly more specific and wins for compiled
# tapes. On the constants path we always pass `DI.Constant(eval.f)` plus the
# `N` context constants — `N == 0` collapses the `map` splat to nothing.
@inline _di_value_and_gradient(c::DICache{:closure}, ad, x, _) =
    DI.value_and_gradient(c.target, c.gradient_prep, ad, x)
@inline _di_value_and_gradient(c::DICache, ad, x, eval) = DI.value_and_gradient(
    c.target,
    c.gradient_prep,
    ad,
    x,
    DI.Constant(eval.f),
    map(DI.Constant, eval.context)...,
)

@inline _di_value_and_jacobian(c::DICache{:closure}, ad, x, _) =
    DI.value_and_jacobian(c.target, c.jacobian_prep, ad, x)
@inline _di_value_and_jacobian(c::DICache, ad, x, eval) = DI.value_and_jacobian(
    c.target,
    c.jacobian_prep,
    ad,
    x,
    DI.Constant(eval.f),
    map(DI.Constant, eval.context)...,
)

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
