module AbstractPPLForwardDiffExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems:
    _assert_gradient_output,
    _assert_jacobian_output,
    _assert_namedtuple_shape,
    _assert_supported_output,
    _is_scalar_output
using AbstractPPL.Utils: flatten_to!!, unflatten_to!!
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff

struct ForwardDiffPrepared{E,F,GC,GR,JC,JR} <: AbstractPPL.ADProblems.AbstractPrepared
    evaluator::E
    f::F
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
    evaluator = AbstractPPL.ADProblems.NamedTupleEvaluator{check_dims}(raw, values)
    # Hand ForwardDiff an unchecked wrapper: the flattened input is reconstructed
    # with Dual-typed fields during tracing, which would fail the exact-type check.
    inner = AbstractPPL.ADProblems.NamedTupleEvaluator{false}(raw, values)
    x = flatten_to!!(nothing, values)
    f = let inner = inner, values = values
        x -> inner(unflatten_to!!(values, x))
    end
    y = f(x)
    _assert_gradient_output(y)
    gradient_result = ForwardDiff.DiffResults.MutableDiffResult(y, (similar(x),))
    gradient_config = _forwarddiff_config(ForwardDiff.GradientConfig, adtype, f, x)
    return ForwardDiffPrepared(
        evaluator, f, gradient_config, gradient_result, nothing, nothing
    )
end

function AbstractPPL.prepare(
    adtype::AutoForwardDiff, problem, x::AbstractVector{<:Real}; check_dims::Bool=true
)
    raw = AbstractPPL.prepare(problem, x)
    length(x) == 0 && return AbstractPPL.ADProblems.VectorEvaluator{check_dims}(raw, 0)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator{check_dims}(raw, length(x))
    # Hand ForwardDiff an unchecked wrapper so the per-call dim check does not
    # land in the dual-number hot path; user-visible `prepared(x)` still goes
    # through `evaluator` (whose `check_dims` honors the caller's request).
    f = AbstractPPL.ADProblems.VectorEvaluator{false}(raw, length(x))
    y = f(x)
    _assert_supported_output(y)
    if _is_scalar_output(y)
        gradient_config = _forwarddiff_config(ForwardDiff.GradientConfig, adtype, f, x)
        gradient_result = ForwardDiff.DiffResults.MutableDiffResult(y, (similar(x),))
        return ForwardDiffPrepared(
            evaluator, f, gradient_config, gradient_result, nothing, nothing
        )
    else
        _assert_jacobian_output(y)
        jacobian_config = _forwarddiff_config(ForwardDiff.JacobianConfig, adtype, f, x)
        jacobian_result = ForwardDiff.DiffResults.JacobianResult(similar(y), x)
        return ForwardDiffPrepared(
            evaluator, f, nothing, nothing, jacobian_config, jacobian_result
        )
    end
end

@inline function AbstractPPL.value_and_gradient(
    p::ForwardDiffPrepared{<:AbstractPPL.ADProblems.NamedTupleEvaluator}, values::NamedTuple
)
    _assert_namedtuple_shape(p.evaluator, values)
    p.gradient_config === nothing &&
        throw(ArgumentError("`value_and_gradient` requires a scalar-valued function."))
    x = flatten_to!!(nothing, values)
    # `Val(false)`: skip ForwardDiff's tag check; `p.gradient_config` is already bound to `p.f`.
    ForwardDiff.gradient!(p.gradient_result, p.f, x, p.gradient_config, Val(false))
    val = ForwardDiff.DiffResults.value(p.gradient_result)
    return (
        val,
        unflatten_to!!(
            p.evaluator.inputspec, ForwardDiff.DiffResults.gradient(p.gradient_result)
        ),
    )
end

@inline function AbstractPPL.value_and_gradient(
    p::ForwardDiffPrepared{<:AbstractPPL.ADProblems.VectorEvaluator},
    x::AbstractVector{<:Real},
)
    p.gradient_config === nothing &&
        throw(ArgumentError("`value_and_gradient` requires a scalar-valued function."))
    ForwardDiff.gradient!(p.gradient_result, p.f, x, p.gradient_config, Val(false))
    val = ForwardDiff.DiffResults.value(p.gradient_result)
    grad = copy(ForwardDiff.DiffResults.gradient(p.gradient_result))
    return (val, grad)
end

@inline function AbstractPPL.value_and_jacobian(
    p::ForwardDiffPrepared{<:AbstractPPL.ADProblems.VectorEvaluator},
    x::AbstractVector{<:Real},
)
    p.jacobian_config === nothing &&
        throw(ArgumentError("`value_and_jacobian` requires a vector-valued function."))
    ForwardDiff.jacobian!(p.jacobian_result, p.f, x, p.jacobian_config, Val(false))
    return (
        copy(ForwardDiff.DiffResults.value(p.jacobian_result)),
        copy(ForwardDiff.DiffResults.jacobian(p.jacobian_result)),
    )
end

function AbstractPPL.ADProblems._supports_gradient(
    ::ForwardDiffPrepared{
        <:AbstractPPL.ADProblems.VectorEvaluator,<:Any,<:Any,<:Any,Nothing,Nothing
    },
)
    return true
end

end # module
