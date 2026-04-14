module AbstractPPLForwardDiffExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff

struct ForwardDiffPrepared{E,F,C,R,P}
    evaluator::E
    f_vec::F
    config::C
    result::R
    prototype::P
    dim::Int
end

AbstractPPL.capabilities(::Type{<:ForwardDiffPrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::ForwardDiffPrepared) = p.dim

function (p::ForwardDiffPrepared)(values::NamedTuple)
    return p.evaluator(values)
end

function (p::ForwardDiffPrepared)(x::AbstractVector)
    return p.f_vec(x)
end

function AbstractPPL.prepare(
    ::AutoForwardDiff, problem, prototype::NamedTuple
)
    evaluator = AbstractPPL.prepare(problem, prototype)
    x0 = AbstractPPL.flatten_to_vec(prototype)
    f_vec = let evaluator = evaluator, prototype = prototype
        x -> evaluator(AbstractPPL.unflatten_from_vec(prototype, x))
    end
    cfg = ForwardDiff.GradientConfig(f_vec, x0)
    grad_buf = similar(x0)
    result = ForwardDiff.DiffResults.MutableDiffResult(zero(eltype(x0)), (grad_buf,))
    return ForwardDiffPrepared(evaluator, f_vec, cfg, result, prototype, length(x0))
end

function AbstractPPL.value_and_gradient(p::ForwardDiffPrepared, values::NamedTuple)
    x = AbstractPPL.flatten_to_vec(values)
    ForwardDiff.gradient!(p.result, p.f_vec, x, p.config)
    val = ForwardDiff.DiffResults.value(p.result)
    dx = ForwardDiff.DiffResults.gradient(p.result)
    grad_nt = AbstractPPL.unflatten_from_vec(p.prototype, dx)
    return (val, grad_nt)
end

end # module
