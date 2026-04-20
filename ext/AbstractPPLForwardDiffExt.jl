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

function AbstractPPL.prepare(::AutoForwardDiff, problem, values::NamedTuple)
    evaluator = AbstractPPL.ADProblems.NamedTupleEvaluator(
        AbstractPPL.prepare(problem, values), values
    )
    x = flatten_to!!(nothing, values)
    f = let evaluator = evaluator, values = values
        x -> evaluator(unflatten_to!!(values, x))
    end
    result = ForwardDiff.DiffResults.MutableDiffResult(zero(eltype(x)), (similar(x),))
    cfg = ForwardDiff.GradientConfig(f, x)
    return ForwardDiffPrepared(evaluator, f, cfg, result)
end

function AbstractPPL.prepare(::AutoForwardDiff, problem, x::AbstractVector{<:AbstractFloat})
    evaluator = AbstractPPL.ADProblems.VectorEvaluator(
        AbstractPPL.prepare(problem, x), length(x)
    )
    cfg = ForwardDiff.GradientConfig(evaluator, x)
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
    ForwardDiff.gradient!(p.result, p.f, x, p.config)
    val = ForwardDiff.DiffResults.value(p.result)
    grad = copy(ForwardDiff.DiffResults.gradient(p.result))
    return (val, unflatten_to!!(p.evaluator.inputspec, grad))
end

@inline function AbstractPPL.value_and_gradient(
    p::ForwardDiffPrepared{<:AbstractPPL.ADProblems.VectorEvaluator},
    x::AbstractVector{<:AbstractFloat},
)
    ForwardDiff.gradient!(p.result, p.f, x, p.config)
    val = ForwardDiff.DiffResults.value(p.result)
    grad = copy(ForwardDiff.DiffResults.gradient(p.result))
    return (val, grad)
end

end # module
