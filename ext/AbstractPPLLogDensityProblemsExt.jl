module AbstractPPLLogDensityProblemsExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems: Prepared, VectorEvaluator
using LogDensityProblems: LogDensityProblems

LogDensityProblems.logdensity(p::Prepared, x) = p(x)
LogDensityProblems.logdensity(e::VectorEvaluator, x) = e(x)

LogDensityProblems.dimension(p::Prepared) = LogDensityProblems.dimension(p.evaluator)
LogDensityProblems.dimension(e::VectorEvaluator) = e.dim

# `Prepared` is the AD-aware shape, so it always advertises gradient capability.
LogDensityProblems.capabilities(::Type{<:Prepared}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.capabilities(p::Prepared) = LogDensityProblems.capabilities(typeof(p))

# A bare `VectorEvaluator` is the no-AD shape; only `Prepared` advertises gradient.
function LogDensityProblems.capabilities(::Type{<:VectorEvaluator})
    return LogDensityProblems.LogDensityOrder{0}()
end
LogDensityProblems.capabilities(e::VectorEvaluator) = LogDensityProblems.capabilities(typeof(e))

function LogDensityProblems.logdensity_and_gradient(p::Prepared, x)
    val, grad = AbstractPPL.value_and_gradient!!(p, x)
    return (val, copy(grad))
end

end # module
