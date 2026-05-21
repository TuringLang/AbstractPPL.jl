module AbstractPPLDifferentiationInterfaceExt

using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators: Evaluators, Prepared, VectorEvaluator, _ad_output_arity
using ADTypes: AbstractADType, AutoReverseDiff
using DifferentiationInterface: DifferentiationInterface as DI

# AD target used by every DI cache. `Vararg{Any,N}` with a free `N` forces
# specialization on the trailing arity (a bare `Vararg{Any}` would skip it).
# DI invokes this as `_call_evaluator(x, f, c1, …, cN)` on the constants path,
# and as `_call_evaluator(x, evaluator)` (via `Fix2`) on the closure path —
# empty `ctx` then makes the splat a no-op.
@inline _call_evaluator(x, f::F, ctx::Vararg{Any,N}) where {F,N} = f(x, ctx...)

# `Mode` tags the call shape:
#   * `:closure` — compiled-tape ReverseDiff: target is a `Fix2` closure; the
#                  AD call passes **0** `DI.Constant`s.
#   * `N::Int`   — constants path: `N == length(evaluator.context)`; the AD
#                  call passes **N + 1** `DI.Constant`s (`f` plus the `N`
#                  context values).
# Encoding `Mode` in each cache type resolves the closure-vs-constants dispatch
# in `_di_value_and_*` at compile time without a runtime branch.

# `Nothing` in the prep slot flags the empty-input cache (DI prep paths fail
# on length-0 input, e.g. ForwardDiff `BoundsError`). Hot paths dispatch on the
# `Nothing` parameter to short-circuit before any DI call. Same convention for
# `DIJacobianCache` and `DIHessianCache` below.
struct DIGradientCache{Mode,F,GP}
    target::F
    gradient_prep::GP
    function DIGradientCache(target::F, gp::GP, ::Val{Mode}) where {Mode,F,GP}
        return new{Mode,F,GP}(target, gp)
    end
end

struct DIJacobianCache{Mode,F,JP}
    target::F
    jacobian_prep::JP
    function DIJacobianCache(target::F, jp::JP, ::Val{Mode}) where {Mode,F,JP}
        return new{Mode,F,JP}(target, jp)
    end
end

# Order=2 (scalar-output). `grad_buf` / `hess_buf` are caller-owned output
# buffers handed to `DI.value_gradient_and_hessian!`; the returned arrays alias
# them (`!!` contract).
struct DIHessianCache{Mode,F,GP,HP,G,H}
    target::F
    gradient_prep::GP
    hessian_prep::HP
    grad_buf::G
    hess_buf::H
    function DIHessianCache(
        target::F, gp::GP, hp::HP, g::G, h::H, ::Val{Mode}
    ) where {Mode,F,GP,HP,G,H}
        return new{Mode,F,GP,HP,G,H}(target, gp, hp, g, h)
    end
end

# Compiled ReverseDiff only reuses a compiled tape on the one-argument path;
# `DI.Constant` deactivates tape recording, so close the evaluator into the
# target and call DI without constants. Context (if any) is captured inside
# the evaluator closure rather than lowered out — the lowered path would also
# require a closure here, so the wrapper cost is unavoidable for compiled tapes.
#
# `_di_call_shape` returns `(target, mode, constants)`. For the closure path
# `constants == ()` and the splat at every prep/call site collapses to nothing,
# letting prep and call sites share one shape regardless of mode.
function _di_call_shape(::AutoReverseDiff{true}, evaluator)
    return Base.Fix2(_call_evaluator, evaluator), Val(:closure), ()
end
function _di_call_shape(::AbstractADType, evaluator)
    return _call_evaluator,
    Val(length(evaluator.context)),
    (DI.Constant(evaluator.f), map(DI.Constant, evaluator.context)...)
end

# `SecondOrder` doesn't define gradient prep; per DI's contract the inner
# adtype is the one used for the first derivative.
@inline _gradient_adtype(adtype::AbstractADType) = adtype
@inline _gradient_adtype(adtype::DI.SecondOrder) = DI.inner(adtype)

function _prepare_di(prep::F, adtype, x, evaluator) where {F}
    target, mode, constants = _di_call_shape(adtype, evaluator)
    return target, prep(target, adtype, x, constants...), mode
end

function AbstractPPL.prepare(
    adtype::AbstractADType,
    problem,
    x::AbstractVector{<:Real};
    check_dims::Bool=true,
    context::Tuple=(),
    order::Int=1,
)
    Evaluators._validate_ad_order(order)
    evaluator = AbstractPPL.prepare(problem, x; check_dims, context)::VectorEvaluator
    arity = _ad_output_arity(evaluator(x))
    mode_empty = Val(length(context))
    if order == 2
        arity === :scalar || Evaluators._throw_hessian_needs_scalar()
        if length(x) == 0
            cache = DIHessianCache(
                _call_evaluator, nothing, nothing, nothing, nothing, mode_empty
            )
            return Prepared(adtype, evaluator, cache, Val(2))
        end
        # Build both gradient and Hessian preps against the same target so
        # `value_and_gradient!!` on the order=2 prep skips the O(n²) Hessian
        # cost. Sharing the target matters for compiled-tape ReverseDiff —
        # two `Fix2` instances may not be interchangeable in DI.
        target, mode, constants = _di_call_shape(adtype, evaluator)
        gradient_prep = DI.prepare_gradient(
            target, _gradient_adtype(adtype), x, constants...
        )
        hessian_prep = DI.prepare_hessian(target, adtype, x, constants...)
        # Buffers pre-allocated from `x`: hot path is zero-allocation on the
        # gradient/Hessian outputs, returned arrays alias these slots.
        cache = DIHessianCache(
            target,
            gradient_prep,
            hessian_prep,
            similar(x),
            similar(x, length(x), length(x)),
            mode,
        )
        return Prepared(adtype, evaluator, cache, Val(2))
    end
    if length(x) == 0
        cache = if arity === :scalar
            DIGradientCache(_call_evaluator, nothing, mode_empty)
        else
            DIJacobianCache(_call_evaluator, nothing, mode_empty)
        end
        return Prepared(adtype, evaluator, cache)
    end
    if arity === :scalar
        target, gradient_prep, mode = _prepare_di(DI.prepare_gradient, adtype, x, evaluator)
        return Prepared(adtype, evaluator, DIGradientCache(target, gradient_prep, mode))
    end
    target, jacobian_prep, mode = _prepare_di(DI.prepare_jacobian, adtype, x, evaluator)
    return Prepared(adtype, evaluator, DIJacobianCache(target, jacobian_prep, mode))
end

# Hot-path dispatch is by cache type + `Mode` (closure vs constants), both
# resolved at compile time. On the constants path we always pass
# `DI.Constant(eval.f)` plus the `N` context constants — `N == 0` collapses
# the `map` splat to nothing.
const _GradientCapable = Union{DIGradientCache,DIHessianCache}

@inline _di_value_and_gradient(c::Union{DIGradientCache{:closure},DIHessianCache{:closure}}, ad, x, _) = DI.value_and_gradient(
    c.target, c.gradient_prep, _gradient_adtype(ad), x
)
@inline _di_value_and_gradient(c::_GradientCapable, ad, x, eval) = DI.value_and_gradient(
    c.target,
    c.gradient_prep,
    _gradient_adtype(ad),
    x,
    DI.Constant(eval.f),
    map(DI.Constant, eval.context)...,
)

@inline _di_value_and_jacobian(c::DIJacobianCache{:closure}, ad, x, _) = DI.value_and_jacobian(
    c.target, c.jacobian_prep, ad, x
)
@inline _di_value_and_jacobian(c::DIJacobianCache, ad, x, eval) = DI.value_and_jacobian(
    c.target, c.jacobian_prep, ad, x, DI.Constant(eval.f), map(DI.Constant, eval.context)...
)

@inline _di_value_gradient_and_hessian(c::DIHessianCache{:closure}, ad, x, _) = DI.value_gradient_and_hessian!(
    c.target, c.grad_buf, c.hess_buf, c.hessian_prep, ad, x
)
@inline _di_value_gradient_and_hessian(c::DIHessianCache, ad, x, eval) = DI.value_gradient_and_hessian!(
    c.target,
    c.grad_buf,
    c.hess_buf,
    c.hessian_prep,
    ad,
    x,
    DI.Constant(eval.f),
    map(DI.Constant, eval.context)...,
)

# `value_and_gradient!!`: works on both `DIGradientCache` (order=1 scalar) and
# `DIHessianCache` (order=2). Empty-input caches carry `gradient_prep::Nothing`
# and dispatch to the short-circuit method below; vector-output caches reject.
@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{
        <:AbstractADType,
        <:VectorEvaluator,
        <:Union{DIGradientCache{<:Any,<:Any,Nothing},DIHessianCache{<:Any,<:Any,Nothing}},
    },
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    return (p.evaluator(x), T[])
end

@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:AbstractADType,<:VectorEvaluator,<:_GradientCapable}, x::AbstractVector{T}
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    return _di_value_and_gradient(p.cache, p.adtype, x, p.evaluator)
end

@inline function AbstractPPL.value_and_gradient!!(
    ::Prepared{<:AbstractADType,<:VectorEvaluator,<:DIJacobianCache},
    ::AbstractVector{<:Real},
)
    return Evaluators._throw_gradient_needs_scalar()
end

@inline function AbstractPPL.value_and_jacobian!!(
    p::Prepared{<:AbstractADType,<:VectorEvaluator,<:DIJacobianCache{<:Any,<:Any,Nothing}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    val = p.evaluator(x)
    return (val, similar(x, length(val), 0))
end

@inline function AbstractPPL.value_and_jacobian!!(
    p::Prepared{<:AbstractADType,<:VectorEvaluator,<:DIJacobianCache}, x::AbstractVector{T}
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    return _di_value_and_jacobian(p.cache, p.adtype, x, p.evaluator)
end

@inline function AbstractPPL.value_and_jacobian!!(
    ::Prepared{<:AbstractADType,<:VectorEvaluator,<:_GradientCapable},
    ::AbstractVector{<:Real},
)
    return Evaluators._throw_jacobian_needs_vector()
end

@inline function AbstractPPL.value_gradient_and_hessian!!(
    p::Prepared{
        <:AbstractADType,<:VectorEvaluator,<:DIHessianCache{<:Any,<:Any,<:Any,Nothing}
    },
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    return (p.evaluator(x), T[], similar(x, 0, 0))
end

@inline function AbstractPPL.value_gradient_and_hessian!!(
    p::Prepared{<:AbstractADType,<:VectorEvaluator,<:DIHessianCache}, x::AbstractVector{T}
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    return _di_value_gradient_and_hessian(p.cache, p.adtype, x, p.evaluator)
end

@inline function AbstractPPL.value_gradient_and_hessian!!(
    ::Prepared{<:AbstractADType,<:VectorEvaluator,<:Union{DIGradientCache,DIJacobianCache}},
    ::AbstractVector{<:Real},
)
    return Evaluators._throw_hessian_needs_order_2_prep()
end

end # module
