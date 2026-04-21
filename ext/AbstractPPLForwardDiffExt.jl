module AbstractPPLForwardDiffExt

using AbstractPPL: AbstractPPL, DerivativeOrder
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
    typeof(values) === typeof(p.evaluator.inputspec) || throw(
        ArgumentError(
            "Expected the same NamedTuple structure that was used to prepare this evaluator.",
        ),
    )
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

function AbstractPPL.prepare(adtype::AutoForwardDiff, problem, values::NamedTuple)
    evaluator = AbstractPPL.ADProblems.NamedTupleEvaluator(
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
    adtype::AutoForwardDiff, problem, x::AbstractVector{<:AbstractFloat}
)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator(
        AbstractPPL.prepare(problem, x), length(x)
    )
    cfg = _forwarddiff_config(adtype, evaluator, x)
    grad_buf = similar(x)
    result = ForwardDiff.DiffResults.MutableDiffResult(zero(eltype(x)), (grad_buf,))
    return ForwardDiffPrepared(evaluator, evaluator, cfg, result)
end

@inline function AbstractPPL.value_and_gradient(
    p::ForwardDiffPrepared{<:AbstractPPL.ADProblems.NamedTupleEvaluator}, values::NamedTuple
)
    typeof(values) === typeof(p.evaluator.inputspec) || throw(
        ArgumentError(
            "Expected the same NamedTuple structure that was used to prepare this evaluator.",
        ),
    )
    x = flatten_to!!(nothing, values)
    # `Val(false)` skips ForwardDiff's tag check; we built `p.config` from `p.f`
    # in `_forwarddiff_config`, so caller-supplied custom tags must be honored.
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
