module AbstractPPLForwardDiffExt

using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators: Evaluators, Prepared, VectorEvaluator, _ad_output_arity
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff
using DiffResults: DiffResults

# `AutoForwardDiff{CS}` carries the chunk size as a type parameter; `nothing`
# defers the choice to ForwardDiff.
_fd_chunk(::AutoForwardDiff{nothing}, x) = ForwardDiff.Chunk(x)
_fd_chunk(::AutoForwardDiff{CS}, _) where {CS} = ForwardDiff.Chunk{CS}()

# Three cache types so arity (scalar/vector) and order (1/2) are encoded in the
# type and dispatch resolves the hot path without runtime branching. The stored
# `result` aliases the arrays returned by `value_and_*!!`, per the `!!` contract.
struct FDGradientCache{R,C}
    result::R
    config::C
end

struct FDJacobianCache{R,C}
    result::R
    config::C
end

# `gradient_result` / `gradient_config` are kept alongside the Hessian cache so
# `value_and_gradient!!` on an order=2 prep skips the O(n²) Hessian work.
struct FDHessianCache{R,C,GR,GC}
    result::R
    config::C
    gradient_result::GR
    gradient_config::GC
end

"""
    prepare(adtype::AutoForwardDiff, problem, x; check_dims=true, context::Tuple=(), order=1)

Prepare a ForwardDiff gradient, Jacobian, or Hessian evaluator for a vector
input. `order=1` (default) picks gradient/Jacobian by output arity; `order=2`
builds Hessian machinery and requires a scalar-valued problem. `context` and
`check_dims` follow the base `prepare` contract.
"""
function AbstractPPL.prepare(
    adtype::AutoForwardDiff,
    problem,
    x::AbstractVector{<:Real};
    check_dims::Bool=true,
    context::Tuple=(),
    order::Int=1,
)
    Evaluators._validate_ad_order(order)
    evaluator = AbstractPPL.prepare(problem, x; check_dims, context)::VectorEvaluator
    arity = _ad_output_arity(evaluator(x))
    chunk = _fd_chunk(adtype, x)

    if order == 2
        arity === :scalar || Evaluators._throw_hessian_needs_scalar()
        length(x) == 0 && return Prepared(
            adtype,
            evaluator,
            FDHessianCache(nothing, nothing, nothing, nothing),
            Val(2),
        )
        hess_result = DiffResults.MutableDiffResult(
            zero(eltype(x)), (similar(x), similar(x, length(x), length(x)))
        )
        hess_config = ForwardDiff.HessianConfig(
            _fd_target(evaluator), hess_result, x, chunk
        )
        grad_result = DiffResults.MutableDiffResult(zero(eltype(x)), (similar(x),))
        grad_config = ForwardDiff.GradientConfig(_fd_target(evaluator), x, chunk)
        cache = FDHessianCache(hess_result, hess_config, grad_result, grad_config)
        return Prepared(adtype, evaluator, cache, Val(2))
    end

    if arity === :scalar
        length(x) == 0 &&
            return Prepared(adtype, evaluator, FDGradientCache(nothing, nothing))
        result = DiffResults.MutableDiffResult(zero(eltype(x)), (similar(x),))
        config = ForwardDiff.GradientConfig(_fd_target(evaluator), x, chunk)
        return Prepared(adtype, evaluator, FDGradientCache(result, config))
    else
        length(x) == 0 &&
            return Prepared(adtype, evaluator, FDJacobianCache(nothing, nothing))
        y = evaluator(x)
        result = DiffResults.MutableDiffResult(
            similar(y), (similar(y, length(y), length(x)),)
        )
        config = ForwardDiff.JacobianConfig(_fd_target(evaluator), x, chunk)
        return Prepared(adtype, evaluator, FDJacobianCache(result, config))
    end
end

# ForwardDiff's `*Config` keys its `Tag` on the *type* of the target, so
# constructing a fresh `Fix2` per hot-path call is free — the type matches the
# one captured in the config at prep time.
@inline _fd_target(e::VectorEvaluator) = Base.Fix2(_fd_call, e)
@inline _fd_call(x, e::VectorEvaluator) = e.f(x, e.context...)

@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:FDGradientCache{Nothing}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    return (p.evaluator(x), T[])
end

@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:FDGradientCache},
    x::AbstractVector{<:Real},
)
    Evaluators._check_ad_input(p.evaluator, x)
    ForwardDiff.gradient!(p.cache.result, _fd_target(p.evaluator), x, p.cache.config)
    return (DiffResults.value(p.cache.result), DiffResults.gradient(p.cache.result))
end

# Arity-mismatch rejections live on dedicated cache types so dispatch resolves
# the failure mode at compile time.
@inline function AbstractPPL.value_and_gradient!!(
    ::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:FDJacobianCache},
    ::AbstractVector{<:Real},
)
    return Evaluators._throw_gradient_needs_scalar()
end

@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:FDHessianCache{Nothing}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    return (p.evaluator(x), T[])
end

@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:FDHessianCache},
    x::AbstractVector{<:Real},
)
    Evaluators._check_ad_input(p.evaluator, x)
    ForwardDiff.gradient!(
        p.cache.gradient_result, _fd_target(p.evaluator), x, p.cache.gradient_config
    )
    return (
        DiffResults.value(p.cache.gradient_result),
        DiffResults.gradient(p.cache.gradient_result),
    )
end

@inline function AbstractPPL.value_and_jacobian!!(
    ::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:Union{FDGradientCache,FDHessianCache}},
    ::AbstractVector{<:Real},
)
    return Evaluators._throw_jacobian_needs_vector()
end

@inline function AbstractPPL.value_and_jacobian!!(
    p::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:FDJacobianCache{Nothing}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    val = p.evaluator(x)
    return (val, similar(x, length(val), 0))
end

@inline function AbstractPPL.value_and_jacobian!!(
    p::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:FDJacobianCache},
    x::AbstractVector{<:Real},
)
    Evaluators._check_ad_input(p.evaluator, x)
    ForwardDiff.jacobian!(p.cache.result, _fd_target(p.evaluator), x, p.cache.config)
    return (DiffResults.value(p.cache.result), DiffResults.jacobian(p.cache.result))
end

@inline function AbstractPPL.value_gradient_and_hessian!!(
    ::Prepared{
        <:AutoForwardDiff,<:VectorEvaluator,<:Union{FDGradientCache,FDJacobianCache}
    },
    ::AbstractVector{<:Real},
)
    return Evaluators._throw_hessian_needs_order_2_prep()
end

@inline function AbstractPPL.value_gradient_and_hessian!!(
    p::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:FDHessianCache{Nothing}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    return (p.evaluator(x), T[], similar(x, 0, 0))
end

@inline function AbstractPPL.value_gradient_and_hessian!!(
    p::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:FDHessianCache},
    x::AbstractVector{<:Real},
)
    Evaluators._check_ad_input(p.evaluator, x)
    ForwardDiff.hessian!(p.cache.result, _fd_target(p.evaluator), x, p.cache.config)
    return (
        DiffResults.value(p.cache.result),
        DiffResults.gradient(p.cache.result),
        DiffResults.hessian(p.cache.result),
    )
end

end # module
