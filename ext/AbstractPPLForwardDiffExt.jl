module AbstractPPLForwardDiffExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using AbstractPPL.ADProblems: _assert_namedtuple_shape
using AbstractPPL.Utils: flatten_to!!, unflatten_to!!
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff

struct ForwardDiffPrepared{E,F,C,R}
    evaluator::E
    f::F
    config::C
    result::R
end

AbstractPPL.capabilities(::Type{<:ForwardDiffPrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::ForwardDiffPrepared) = AbstractPPL.dimension(p.evaluator)

function (p::ForwardDiffPrepared)(x::AbstractVector{<:Integer})
    throw(MethodError(p, (x,)))
end

function (p::ForwardDiffPrepared{<:AbstractPPL.ADProblems.NamedTupleEvaluator})(
    values::NamedTuple
)
    _assert_namedtuple_shape(p.evaluator, values)
    return p.evaluator(values)
end

(p::ForwardDiffPrepared)(x) = p.evaluator(x)

function _forwarddiff_chunk(::AutoForwardDiff{nothing}, x)
    return ForwardDiff.Chunk(x)
end
function _forwarddiff_chunk(::AutoForwardDiff{chunksize}, x) where {chunksize}
    return ForwardDiff.Chunk{chunksize}()
end

function _forwarddiff_tag(adtype::AutoForwardDiff, f, x)
    return adtype.tag === nothing ? ForwardDiff.Tag(f, eltype(x)) : adtype.tag
end

function _forwarddiff_config(adtype::AutoForwardDiff, f, x)
    return ForwardDiff.GradientConfig(
        f, x, _forwarddiff_chunk(adtype, x), _forwarddiff_tag(adtype, f, x)
    )
end

function AbstractPPL.prepare(
    adtype::AutoForwardDiff, problem, values::NamedTuple; check_dims::Bool=true
)
    evaluator = AbstractPPL.ADProblems.NamedTupleEvaluator{check_dims}(
        AbstractPPL.prepare(problem, values), values
    )
    x = flatten_to!!(nothing, values)
    f = let evaluator = evaluator, values = values
        x -> evaluator(unflatten_to!!(values, x))
    end
    result = ForwardDiff.DiffResults.MutableDiffResult(zero(eltype(x)), (similar(x),))
    cfg = _forwarddiff_config(adtype, f, x)
    return ForwardDiffPrepared(evaluator, f, cfg, result)
end

function AbstractPPL.prepare(
    adtype::AutoForwardDiff,
    problem,
    x::AbstractVector{<:AbstractFloat};
    check_dims::Bool=true,
)
    raw = AbstractPPL.prepare(problem, x)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator{check_dims}(raw, length(x))
    # Hand ForwardDiff an unchecked wrapper so the per-call dim check does not
    # land in the dual-number hot path; user-visible `prepared(x)` still goes
    # through `evaluator` (whose `check_dims` honors the caller's request).
    f = AbstractPPL.ADProblems.VectorEvaluator{false}(raw, length(x))
    cfg = _forwarddiff_config(adtype, f, x)
    grad_buf = similar(x)
    result = ForwardDiff.DiffResults.MutableDiffResult(zero(eltype(x)), (grad_buf,))
    return ForwardDiffPrepared(evaluator, f, cfg, result)
end

@inline function AbstractPPL.value_and_gradient(
    p::ForwardDiffPrepared{<:AbstractPPL.ADProblems.NamedTupleEvaluator}, values::NamedTuple
)
    _assert_namedtuple_shape(p.evaluator, values)
    x = flatten_to!!(nothing, values)
    # `Val(false)`: skip ForwardDiff's tag check; `p.config` is already bound to `p.f`.
    ForwardDiff.gradient!(p.result, p.f, x, p.config, Val(false))
    val = ForwardDiff.DiffResults.value(p.result)
    grad = copy(ForwardDiff.DiffResults.gradient(p.result))
    return (val, unflatten_to!!(p.evaluator.inputspec, grad))
end

@inline function AbstractPPL.value_and_gradient(
    p::ForwardDiffPrepared{<:AbstractPPL.ADProblems.VectorEvaluator},
    x::AbstractVector{<:AbstractFloat},
)
    ForwardDiff.gradient!(p.result, p.f, x, p.config, Val(false))
    val = ForwardDiff.DiffResults.value(p.result)
    grad = copy(ForwardDiff.DiffResults.gradient(p.result))
    return (val, grad)
end

end # module
