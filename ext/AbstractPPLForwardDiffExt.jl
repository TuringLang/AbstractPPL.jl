module AbstractPPLForwardDiffExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems:
    Prepared,
    VectorEvaluator,
    NamedTupleEvaluator,
    _assert_jacobian_output,
    _assert_namedtuple_shape,
    _assert_supported_output,
    _is_scalar_output
using AbstractPPL.Utils: flatten_to!!, unflatten_to!!
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff

struct ForwardDiffCache{F,GC,GR,JC,JR}
    f::F                  # unchecked inner callable for AD tracing
    gradient_config::GC
    gradient_result::GR
    jacobian_config::JC
    jacobian_result::JR
end

function _forwarddiff_chunk(::AutoForwardDiff{nothing}, x)
    return ForwardDiff.Chunk(x)
end
function _forwarddiff_chunk(::AutoForwardDiff{chunksize}, x) where {chunksize}
    return ForwardDiff.Chunk{chunksize}()
end

function _forwarddiff_tag(adtype::AutoForwardDiff, f, x)
    return adtype.tag === nothing ? ForwardDiff.Tag(f, eltype(x)) : adtype.tag
end

function _forwarddiff_config(ConfigType, adtype::AutoForwardDiff, f, x)
    return ConfigType(f, x, _forwarddiff_chunk(adtype, x), _forwarddiff_tag(adtype, f, x))
end

function AbstractPPL.prepare(
    adtype::AutoForwardDiff, problem, values::NamedTuple; check_dims::Bool=true
)
    raw = AbstractPPL.prepare(problem, values)
    evaluator = NamedTupleEvaluator{check_dims}(raw, values)
    # Hand ForwardDiff an unchecked wrapper: the flattened input is reconstructed
    # with Dual-typed fields during tracing, which would fail the exact-type check.
    inner = NamedTupleEvaluator{false}(raw, values)
    x = flatten_to!!(nothing, values)
    f = let inner = inner, values = values
        x -> inner(unflatten_to!!(values, x))
    end
    y = f(x)
    _is_scalar_output(y) || throw(
        ArgumentError(
            "`prepare(::AutoForwardDiff, ...)` for NamedTuple inputs requires a scalar-valued evaluator; got $(typeof(y)).",
        ),
    )
    gradient_result = ForwardDiff.DiffResults.MutableDiffResult(y, (similar(x),))
    gradient_config = _forwarddiff_config(ForwardDiff.GradientConfig, adtype, f, x)
    cache = ForwardDiffCache(f, gradient_config, gradient_result, nothing, nothing)
    return Prepared(adtype, evaluator, cache)
end

function AbstractPPL.prepare(
    adtype::AutoForwardDiff, problem, x::AbstractVector{<:Real}; check_dims::Bool=true
)
    raw = AbstractPPL.prepare(problem, x)
    evaluator = VectorEvaluator{check_dims}(raw, length(x))
    # ForwardDiff has no configuration to build for length-zero arrays; short-circuit
    # in `value_and_gradient!!` / `value_and_jacobian!!` instead.
    length(x) == 0 &&
        return Prepared(adtype, evaluator, ForwardDiffCache(nothing, nothing, nothing, nothing, nothing))
    # Hand ForwardDiff an unchecked wrapper so the per-call dim check does not
    # land in the dual-number hot path; user-visible `prepared(x)` still goes
    # through `evaluator` (whose `check_dims` honors the caller's request).
    f = VectorEvaluator{false}(raw, length(x))
    y = f(x)
    _assert_supported_output(y)
    if _is_scalar_output(y)
        gradient_config = _forwarddiff_config(ForwardDiff.GradientConfig, adtype, f, x)
        gradient_result = ForwardDiff.DiffResults.MutableDiffResult(y, (similar(x),))
        cache = ForwardDiffCache(f, gradient_config, gradient_result, nothing, nothing)
    else
        _assert_jacobian_output(y)
        jacobian_config = _forwarddiff_config(ForwardDiff.JacobianConfig, adtype, f, x)
        jacobian_result = ForwardDiff.DiffResults.JacobianResult(similar(y), x)
        cache = ForwardDiffCache(f, nothing, nothing, jacobian_config, jacobian_result)
    end
    return Prepared(adtype, evaluator, cache)
end

@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{AutoForwardDiff,<:NamedTupleEvaluator,<:ForwardDiffCache}, values::NamedTuple
)
    _assert_namedtuple_shape(p.evaluator, values)
    p.cache.gradient_config === nothing &&
        throw(ArgumentError("`value_and_gradient!!` requires a scalar-valued function."))
    x = flatten_to!!(nothing, values)
    # `Val(false)`: skip ForwardDiff's tag check; config is already bound to `p.cache.f`.
    ForwardDiff.gradient!(p.cache.gradient_result, p.cache.f, x, p.cache.gradient_config, Val(false))
    val = ForwardDiff.DiffResults.value(p.cache.gradient_result)
    return (
        val,
        unflatten_to!!(
            p.evaluator.inputspec, ForwardDiff.DiffResults.gradient(p.cache.gradient_result)
        ),
    )
end

@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{AutoForwardDiff,<:VectorEvaluator,<:ForwardDiffCache}, x::AbstractVector{T}
) where {T<:Real}
    length(x) == 0 && return (p.evaluator(x), T[])
    p.cache.gradient_config === nothing &&
        throw(ArgumentError("`value_and_gradient!!` requires a scalar-valued function."))
    ForwardDiff.gradient!(p.cache.gradient_result, p.cache.f, x, p.cache.gradient_config, Val(false))
    val = ForwardDiff.DiffResults.value(p.cache.gradient_result)
    grad = ForwardDiff.DiffResults.gradient(p.cache.gradient_result)
    return (val, grad)
end

@inline function AbstractPPL.value_and_jacobian!!(
    p::Prepared{AutoForwardDiff,<:VectorEvaluator,<:ForwardDiffCache},
    x::AbstractVector{<:Real},
)
    if length(x) == 0
        val = p.evaluator(x)
        return (val, similar(x, length(val), 0))
    end
    p.cache.jacobian_config === nothing &&
        throw(ArgumentError("`value_and_jacobian!!` requires a vector-valued function."))
    ForwardDiff.jacobian!(p.cache.jacobian_result, p.cache.f, x, p.cache.jacobian_config, Val(false))
    return (
        ForwardDiff.DiffResults.value(p.cache.jacobian_result),
        ForwardDiff.DiffResults.jacobian(p.cache.jacobian_result),
    )
end

end # module
