module AbstractPPLForwardDiffExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using AbstractPPL.Utils: flatten_to!!, unflatten_to!!
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff

struct ForwardDiffPrepared{F,C,R,P}
    evaluator
    f::F
    config::C
    result::R
    inputspec::P
end

AbstractPPL.capabilities(::Type{<:ForwardDiffPrepared}) = DerivativeOrder{1}()

function AbstractPPL.dimension(::ForwardDiffPrepared{<:Any,<:Any,<:Any,<:NamedTuple})
    throw(
        ArgumentError(
            "`dimension` is only available for evaluators prepared with a vector of floating-point numbers.",
        ),
    )
end
function AbstractPPL.dimension(p::ForwardDiffPrepared{<:Any,<:Any,<:Any,<:AbstractVector})
    return length(p.inputspec)
end

function (p::ForwardDiffPrepared{<:Any,<:Any,<:Any,<:NamedTuple})(values::NamedTuple)
    typeof(values) === typeof(p.inputspec) || throw(
        ArgumentError(
            "Expected the same NamedTuple structure that was used to prepare this evaluator.",
        ),
    )
    return p.evaluator(values)
end

function (p::ForwardDiffPrepared{<:Any,<:Any,<:Any,<:AbstractVector})(
    x::AbstractVector{<:Integer}
)
    throw(MethodError(p, (x,)))
end

function (p::ForwardDiffPrepared{<:Any,<:Any,<:Any,<:AbstractVector})(x::AbstractVector)
    length(x) == length(p.inputspec) || throw(
        DimensionMismatch(
            "Expected a vector of length $(length(p.inputspec)), but got length $(length(x)).",
        ),
    )
    return p.evaluator(x)
end

function (p::ForwardDiffPrepared)(x)
    throw(MethodError(p, (x,)))
end

function AbstractPPL.prepare(::AutoForwardDiff, problem, values::NamedTuple)
    evaluator = AbstractPPL.prepare(problem, values)
    x = flatten_to!!(nothing, values)
    f = let evaluator = evaluator, values = values
        x -> evaluator(unflatten_to!!(values, x))
    end
    result = ForwardDiff.DiffResults.MutableDiffResult(zero(eltype(x)), (similar(x),))
    cfg = ForwardDiff.GradientConfig(f, x)
    return ForwardDiffPrepared(evaluator, f, cfg, result, values)
end

function AbstractPPL.prepare(::AutoForwardDiff, problem, x::AbstractVector{<:AbstractFloat})
    evaluator = AbstractPPL.prepare(problem, x)
    f = evaluator
    cfg = ForwardDiff.GradientConfig(f, x)
    grad_buf = similar(x)
    result = ForwardDiff.DiffResults.MutableDiffResult(zero(eltype(x)), (grad_buf,))
    return ForwardDiffPrepared(evaluator, f, cfg, result, x)
end

@inline function AbstractPPL.value_and_gradient(
    p::ForwardDiffPrepared{<:Any,<:Any,<:Any,<:NamedTuple}, values::NamedTuple
)
    typeof(values) === typeof(p.inputspec) || throw(
        ArgumentError(
            "Expected the same NamedTuple structure that was used to prepare this evaluator.",
        ),
    )
    x = flatten_to!!(nothing, values)
    ForwardDiff.gradient!(p.result, p.f, x, p.config)
    val = ForwardDiff.DiffResults.value(p.result)
    grad = copy(ForwardDiff.DiffResults.gradient(p.result))
    return (val, unflatten_to!!(p.inputspec, grad))
end

@inline function AbstractPPL.value_and_gradient(
    p::ForwardDiffPrepared{<:Any,<:Any,<:Any,<:AbstractVector},
    x::AbstractVector{<:AbstractFloat},
)
    length(x) == length(p.inputspec) || throw(
        DimensionMismatch(
            "Expected a vector of length $(length(p.inputspec)), but got length $(length(x)).",
        ),
    )
    ForwardDiff.gradient!(p.result, p.f, x, p.config)
    val = ForwardDiff.DiffResults.value(p.result)
    grad = copy(ForwardDiff.DiffResults.gradient(p.result))
    return (val, grad)
end

end # module
