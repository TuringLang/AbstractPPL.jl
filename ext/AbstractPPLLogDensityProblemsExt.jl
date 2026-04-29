module AbstractPPLLogDensityProblemsExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems: AbstractPrepared, VectorEvaluator
using LogDensityProblems: LogDensityProblems

LogDensityProblems.logdensity(p::AbstractPrepared, x) = p(x)
LogDensityProblems.logdensity(e::VectorEvaluator, x) = e(x)

function LogDensityProblems.dimension(p::AbstractPrepared)
    return LogDensityProblems.dimension(p.evaluator)
end
LogDensityProblems.dimension(e::VectorEvaluator) = e.dim

# `AbstractPrepared` is the AD-aware shape, so it always advertises gradient capability.
function LogDensityProblems.capabilities(::Type{<:AbstractPrepared})
    return LogDensityProblems.LogDensityOrder{1}()
end
function LogDensityProblems.capabilities(p::AbstractPrepared)
    return LogDensityProblems.capabilities(typeof(p))
end

# A bare `VectorEvaluator` is the no-AD shape; only `AbstractPrepared` advertises gradient.
function LogDensityProblems.capabilities(::Type{<:VectorEvaluator})
    return LogDensityProblems.LogDensityOrder{0}()
end
function LogDensityProblems.capabilities(e::VectorEvaluator)
    return LogDensityProblems.capabilities(typeof(e))
end

function LogDensityProblems.logdensity_and_gradient(p::AbstractPrepared, x)
    return AbstractPPL.value_and_gradient(p, x)
end

end # module
