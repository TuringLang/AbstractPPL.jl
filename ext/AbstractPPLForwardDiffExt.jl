module AbstractPPLForwardDiffExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems: _assert_namedtuple_shape, _check_mode, _check_namedtuple_mode
using AbstractPPL.Utils: flatten_to!!, unflatten_to!!
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff

struct ForwardDiffPrepared{Mode,E,F,C,R} <: AbstractPPL.ADProblems.AbstractPrepared{Mode}
    evaluator::E
    f::F
    config::C
    result::R
    function ForwardDiffPrepared{Mode}(
        evaluator::E, f::F, config::C, result::R
    ) where {Mode,E,F,C,R}
        return new{Mode,E,F,C,R}(evaluator, f, config, result)
    end
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
    adtype::AutoForwardDiff,
    problem,
    values::NamedTuple;
    check_dims::Bool=true,
    mode::Symbol=:gradient,
)
    _check_namedtuple_mode(mode)
    raw = AbstractPPL.prepare(problem, values)
    evaluator = AbstractPPL.ADProblems.NamedTupleEvaluator{check_dims}(raw, values)
    # Hand ForwardDiff an unchecked wrapper: the flattened input is reconstructed
    # with Dual-typed fields during tracing, which would fail the exact-type check.
    inner = AbstractPPL.ADProblems.NamedTupleEvaluator{false}(raw, values)
    x = flatten_to!!(nothing, values)
    f = let inner = inner, values = values
        x -> inner(unflatten_to!!(values, x))
    end
    result = ForwardDiff.DiffResults.MutableDiffResult(f(x), (similar(x),))
    cfg = _forwarddiff_config(ForwardDiff.GradientConfig, adtype, f, x)
    return ForwardDiffPrepared{:gradient}(evaluator, f, cfg, result)
end

function AbstractPPL.prepare(
    adtype::AutoForwardDiff,
    problem,
    x::AbstractVector{<:Real};
    check_dims::Bool=true,
    mode::Symbol=:gradient,
)
    _check_mode(mode)
    raw = AbstractPPL.prepare(problem, x)
    length(x) == 0 && return AbstractPPL.ADProblems.VectorEvaluator{check_dims}(raw, 0)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator{check_dims}(raw, length(x))
    # Hand ForwardDiff an unchecked wrapper so the per-call dim check does not
    # land in the dual-number hot path; user-visible `prepared(x)` still goes
    # through `evaluator` (whose `check_dims` honors the caller's request).
    f = AbstractPPL.ADProblems.VectorEvaluator{false}(raw, length(x))
    if mode === :gradient
        cfg = _forwarddiff_config(ForwardDiff.GradientConfig, adtype, f, x)
        result = ForwardDiff.DiffResults.MutableDiffResult(zero(eltype(x)), (similar(x),))
        return ForwardDiffPrepared{:gradient}(evaluator, f, cfg, result)
    else
        # Probe `f` once at prepare-time so we can size the JacobianResult buffer.
        y = f(x)
        y isa AbstractVector || throw(
            ArgumentError(
                "`mode=:jacobian` requires `f(x)` to return an AbstractVector; got $(typeof(y)).",
            ),
        )
        cfg = _forwarddiff_config(ForwardDiff.JacobianConfig, adtype, f, x)
        result = ForwardDiff.DiffResults.JacobianResult(similar(y), x)
        return ForwardDiffPrepared{:jacobian}(evaluator, f, cfg, result)
    end
end

@inline function AbstractPPL.value_and_gradient(
    p::ForwardDiffPrepared{:gradient,<:AbstractPPL.ADProblems.NamedTupleEvaluator},
    values::NamedTuple,
)
    _assert_namedtuple_shape(p.evaluator, values)
    x = flatten_to!!(nothing, values)
    # `Val(false)`: skip ForwardDiff's tag check; `p.config` is already bound to `p.f`.
    ForwardDiff.gradient!(p.result, p.f, x, p.config, Val(false))
    val = ForwardDiff.DiffResults.value(p.result)
    return (
        val,
        unflatten_to!!(p.evaluator.inputspec, ForwardDiff.DiffResults.gradient(p.result)),
    )
end

@inline function AbstractPPL.value_and_gradient(
    p::ForwardDiffPrepared{:gradient,<:AbstractPPL.ADProblems.VectorEvaluator},
    x::AbstractVector{<:Real},
)
    ForwardDiff.gradient!(p.result, p.f, x, p.config, Val(false))
    val = ForwardDiff.DiffResults.value(p.result)
    grad = copy(ForwardDiff.DiffResults.gradient(p.result))
    return (val, grad)
end

@inline function AbstractPPL.value_and_jacobian(
    p::ForwardDiffPrepared{:jacobian,<:AbstractPPL.ADProblems.VectorEvaluator},
    x::AbstractVector{<:Real},
)
    ForwardDiff.jacobian!(p.result, p.f, x, p.config, Val(false))
    val = ForwardDiff.DiffResults.value(p.result)
    jac = copy(ForwardDiff.DiffResults.jacobian(p.result))
    return (copy(val), jac)
end

end # module
