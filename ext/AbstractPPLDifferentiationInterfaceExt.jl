module AbstractPPLDifferentiationInterfaceExt

using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators: Evaluators, Prepared, VectorEvaluator, _ad_output_arity
using ADTypes: AbstractADType, AutoReverseDiff
using DifferentiationInterface: DifferentiationInterface as DI

# AD target used by every `DICache` mode. `Vararg{Any,N}` with a free `N`
# forces specialization on the trailing arity (a bare `Vararg{Any}` would
# skip it). DI invokes this as `_call_evaluator(x, f, c1, …, cN)` on the
# constants path, and as `_call_evaluator(x, evaluator)` (via `Fix2`) on
# the closure path — empty `ctx` then makes the splat a no-op.
@inline _call_evaluator(x, f::F, ctx::Vararg{Any,N}) where {F,N} = f(x, ctx...)

# `Mode` tags the cache shape:
#   * `:closure` — compiled-tape ReverseDiff: target is a `Fix2` closure, the
#                  AD call passes **0** `DI.Constant`s.
#   * `N::Int`   — constants path: `N == length(evaluator.context)`, the AD
#                  call passes **N + 1** `DI.Constant`s (`f` plus the `N`
#                  context values).
# Encoding `Mode` in the type resolves the dispatch in `_di_value_and_*` at
# compile time without a runtime branch.
#
# Single cache for every derivative order. At most one of `gradient_prep`,
# `jacobian_prep`, `hessian_prep` is non-`Nothing` at any time; the hot-path
# methods discriminate via `=== nothing` checks (folded at compile time since
# field types are concrete in each instantiation). `grad_buf` / `hess_buf` are
# non-`Nothing` only for order=2 — caller-owned output buffers handed to
# `DI.value_gradient_and_hessian!`. Returned arrays alias them (`!!` contract).
struct DICache{Mode,F,GP,JP,HP,G,H}
    target::F
    gradient_prep::GP
    jacobian_prep::JP
    hessian_prep::HP
    grad_buf::G
    hess_buf::H
    function DICache{Mode}(
        target::F, gp::GP, jp::JP, hp::HP, g::G, h::H
    ) where {Mode,F,GP,JP,HP,G,H}
        return new{Mode,F,GP,JP,HP,G,H}(target, gp, jp, hp, g, h)
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
    DICache{Mode}(target, gp, jp, nothing, nothing, nothing)

function AbstractPPL.prepare(
    adtype::AbstractADType,
    problem,
    x::AbstractVector{<:Real};
    check_dims::Bool=true,
    context::Tuple=(),
    order::Int=1,
)
    evaluator = AbstractPPL.prepare(problem, x; check_dims, context)::VectorEvaluator
    arity = _ad_output_arity(evaluator(x))
    if order == 2
        arity === :scalar || Evaluators._throw_hessian_needs_scalar()
        if length(x) == 0
            # DI Hessian prep crashes on length-0 input; the AD entry
            # short-circuits before any DI call. `Val(0)` is a non-`Nothing`
            # sentinel for `hessian_prep` so dispatch recognises this as an
            # order=2 prep (mirrors the order=1 empty-input pattern below).
            cache = _wrap_hessian_cache(
                _call_evaluator, Val(0), nothing, nothing, Val(length(context))
            )
            return Prepared(adtype, evaluator, cache)
        end
        target, hessian_prep, mode = _prepare_di(DI.prepare_hessian, adtype, x, evaluator)
        # Buffers pre-allocated from `x` (shape and eltype): the hot path is
        # zero-allocation on the gradient/Hessian outputs, and the returned
        # arrays alias these slots — copy if you need to retain them.
        grad_buf = similar(x)
        hess_buf = similar(x, length(x), length(x))
        cache = _wrap_hessian_cache(target, hessian_prep, grad_buf, hess_buf, mode)
        return Prepared(adtype, evaluator, cache)
    end
    order == 1 || throw(ArgumentError("`order` must be 1 or 2, got $order."))
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

@inline _wrap_hessian_cache(target, hp, g, h, ::Val{Mode}) where {Mode} =
    DICache{Mode}(target, nothing, nothing, hp, g, h)

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
    # Both `=== nothing` branches fold at compile time: each instantiation
    # has concrete field types, so only the relevant branch survives.
    p.cache.hessian_prep === nothing || Evaluators._throw_use_value_gradient_and_hessian()
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
    p.cache.hessian_prep === nothing || Evaluators._throw_use_value_gradient_and_hessian()
    p.cache.jacobian_prep === nothing && Evaluators._throw_jacobian_needs_vector()
    Evaluators._check_ad_input(p.evaluator, x)
    if length(x) == 0
        val = p.evaluator(x)
        return (val, similar(x, length(val), 0))
    end
    return _di_value_and_jacobian(p.cache, p.adtype, x, p.evaluator)
end

# Hessian hot-path dispatch mirrors the gradient/jacobian helpers above:
# `:closure` (compiled-tape) vs constants `Mode`, resolved at compile time.
# Uses DI's in-place variant `value_gradient_and_hessian!` with caller-owned
# buffers; the returned `(val, grad, hess)` aliases `c.grad_buf` / `c.hess_buf`.
@inline _di_value_gradient_and_hessian(c::DICache{:closure}, ad, x, _) =
    DI.value_gradient_and_hessian!(c.target, c.grad_buf, c.hess_buf, c.hessian_prep, ad, x)
@inline _di_value_gradient_and_hessian(c::DICache, ad, x, eval) =
    DI.value_gradient_and_hessian!(
        c.target,
        c.grad_buf,
        c.hess_buf,
        c.hessian_prep,
        ad,
        x,
        DI.Constant(eval.f),
        map(DI.Constant, eval.context)...,
    )

@inline function AbstractPPL.value_gradient_and_hessian!!(
    p::Prepared{<:AbstractADType,<:VectorEvaluator,<:DICache}, x::AbstractVector{T}
) where {T<:Real}
    # Order=1 preps have `hessian_prep === nothing` (compile-folded check).
    p.cache.hessian_prep === nothing && Evaluators._throw_hessian_needs_order_2_prep()
    Evaluators._check_ad_input(p.evaluator, x)
    # Empty-input shortcut — same reasoning as the order=1 path.
    length(x) == 0 && return (p.evaluator(x), T[], similar(x, 0, 0))
    return _di_value_gradient_and_hessian(p.cache, p.adtype, x, p.evaluator)
end

end # module
