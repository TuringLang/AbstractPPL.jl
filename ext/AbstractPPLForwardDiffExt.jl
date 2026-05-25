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

# A user-supplied `adtype.tag` (for nested differentiation) is threaded into the
# `*Config` constructors; `nothing` (the ADTypes default) reproduces
# ForwardDiff's per-constructor default of `Tag(target, eltype(x))`.
@inline _fd_tag(adtype::AutoForwardDiff, target, x) =
    adtype.tag === nothing ? ForwardDiff.Tag(target, eltype(x)) : adtype.tag

# `A::Symbol` ∈ `(:scalar, :vector, :hessian)` encodes both output arity
# (order=1) and order (order=2 ≡ `:hessian`), so dispatch resolves the hot path
# and the arity-mismatch failure modes at compile time without a runtime branch.
# `gradient_result` / `gradient_config` are populated only on `:hessian` caches
# so `value_and_gradient!!` on an order=2 prep skips the O(n²) Hessian work.
# `result::Nothing` is the empty-input sentinel: hot paths dispatch on
# `FDCache{A,Nothing}` to short-circuit before any ForwardDiff call (chunk
# selection `BoundsError`s on length-zero inputs). The stored `result` aliases
# the arrays returned by `value_and_*!!`, per the `!!` contract.
struct FDCache{A,R,C,GR,GC}
    result::R
    config::C
    gradient_result::GR
    gradient_config::GC
    function FDCache{A}(
        result::R, config::C, gradient_result::GR=nothing, gradient_config::GC=nothing
    ) where {A,R,C,GR,GC}
        return new{A,R,C,GR,GC}(result, config, gradient_result, gradient_config)
    end
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
    # Probe the output once: the value classifies arity, and the vector branch
    # reuses it as the Jacobian-result prototype. The base `prepare` contract
    # promises one prep-time call into `problem`.
    y_probe = evaluator(x)
    arity = _ad_output_arity(y_probe)
    chunk = _fd_chunk(adtype, x)
    target = _fd_target(evaluator)
    tag = _fd_tag(adtype, target, x)

    if order == 2
        arity === :scalar || Evaluators._throw_hessian_needs_scalar()
        length(x) == 0 && return Prepared(
            adtype,
            evaluator,
            FDCache{:hessian}(nothing, nothing, nothing, nothing),
            Val(2),
        )
        hess_result = DiffResults.MutableDiffResult(
            zero(eltype(x)), (similar(x), similar(x, length(x), length(x)))
        )
        hess_config = ForwardDiff.HessianConfig(target, hess_result, x, chunk, tag)
        grad_result = DiffResults.MutableDiffResult(zero(eltype(x)), (similar(x),))
        grad_config = ForwardDiff.GradientConfig(target, x, chunk, tag)
        cache = FDCache{:hessian}(hess_result, hess_config, grad_result, grad_config)
        return Prepared(adtype, evaluator, cache, Val(2))
    end

    if arity === :scalar
        length(x) == 0 &&
            return Prepared(adtype, evaluator, FDCache{:scalar}(nothing, nothing))
        result = DiffResults.MutableDiffResult(zero(eltype(x)), (similar(x),))
        config = ForwardDiff.GradientConfig(target, x, chunk, tag)
        return Prepared(adtype, evaluator, FDCache{:scalar}(result, config))
    else
        length(x) == 0 &&
            return Prepared(adtype, evaluator, FDCache{:vector}(nothing, nothing))
        result = DiffResults.MutableDiffResult(
            similar(y_probe), (similar(y_probe, length(y_probe), length(x)),)
        )
        config = ForwardDiff.JacobianConfig(target, x, chunk, tag)
        return Prepared(adtype, evaluator, FDCache{:vector}(result, config))
    end
end

# ForwardDiff's `*Config` keys its `Tag` on the *type* of the target, so
# constructing a fresh `Fix2` per hot-path call is free — the type matches the
# one captured in the config at prep time.
@inline _fd_target(e::VectorEvaluator) = Base.Fix2(_fd_call, e)
@inline _fd_call(x, e::VectorEvaluator) = e.f(x, e.context...)

@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{
        <:AutoForwardDiff,
        <:VectorEvaluator,
        <:Union{FDCache{:scalar,Nothing},FDCache{:hessian,Nothing}},
    },
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    return (p.evaluator(x), T[])
end

@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:FDCache{:scalar}},
    x::AbstractVector{<:Real},
)
    Evaluators._check_ad_input(p.evaluator, x)
    ForwardDiff.gradient!(p.cache.result, _fd_target(p.evaluator), x, p.cache.config)
    return (DiffResults.value(p.cache.result), DiffResults.gradient(p.cache.result))
end

# Order=2 prep also satisfies the order=1 gradient contract via the dedicated
# gradient cache built at prep time — skips the O(n²) Hessian work.
@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:FDCache{:hessian}},
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

# Arity-mismatch rejections live on dedicated cache tags so dispatch resolves
# the failure mode at compile time.
@inline function AbstractPPL.value_and_gradient!!(
    ::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:FDCache{:vector}},
    ::AbstractVector{<:Real},
)
    return Evaluators._throw_gradient_needs_scalar()
end

@inline function AbstractPPL.value_and_jacobian!!(
    ::Prepared{
        <:AutoForwardDiff,<:VectorEvaluator,<:Union{FDCache{:scalar},FDCache{:hessian}}
    },
    ::AbstractVector{<:Real},
)
    return Evaluators._throw_jacobian_needs_vector()
end

@inline function AbstractPPL.value_and_jacobian!!(
    p::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:FDCache{:vector,Nothing}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    val = p.evaluator(x)
    return (val, similar(x, length(val), 0))
end

@inline function AbstractPPL.value_and_jacobian!!(
    p::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:FDCache{:vector}},
    x::AbstractVector{<:Real},
)
    Evaluators._check_ad_input(p.evaluator, x)
    ForwardDiff.jacobian!(p.cache.result, _fd_target(p.evaluator), x, p.cache.config)
    return (DiffResults.value(p.cache.result), DiffResults.jacobian(p.cache.result))
end

@inline function AbstractPPL.value_gradient_and_hessian!!(
    ::Prepared{
        <:AutoForwardDiff,<:VectorEvaluator,<:Union{FDCache{:scalar},FDCache{:vector}}
    },
    ::AbstractVector{<:Real},
)
    return Evaluators._throw_hessian_needs_order_2_prep()
end

@inline function AbstractPPL.value_gradient_and_hessian!!(
    p::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:FDCache{:hessian,Nothing}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    return (p.evaluator(x), T[], similar(x, 0, 0))
end

@inline function AbstractPPL.value_gradient_and_hessian!!(
    p::Prepared{<:AutoForwardDiff,<:VectorEvaluator,<:FDCache{:hessian}},
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
