module AbstractPPLLogDensityProblemsExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems: AbstractPrepared, VectorEvaluator, _supports_gradient
using LogDensityProblems: LogDensityProblems

LogDensityProblems.logdensity(p::AbstractPrepared, x) = p(x)
LogDensityProblems.logdensity(e::VectorEvaluator, x) = e(x)

function LogDensityProblems.dimension(p::AbstractPrepared)
    return LogDensityProblems.dimension(p.evaluator)
end
LogDensityProblems.dimension(e::VectorEvaluator) = e.dim

# Gradient capability is delegated to the `_supports_gradient` trait so that
# only prepared shapes with an actual gradient implementation advertise order 1.
function LogDensityProblems.capabilities(::Type{T}) where {T<:AbstractPrepared}
    return if _supports_gradient(T)
        LogDensityProblems.LogDensityOrder{1}()
    else
        LogDensityProblems.LogDensityOrder{0}()
    end
end
function LogDensityProblems.capabilities(p::AbstractPrepared)
    return LogDensityProblems.capabilities(typeof(p))
end

function LogDensityProblems.capabilities(::Type{T}) where {T<:VectorEvaluator}
    return if _supports_gradient(T)
        LogDensityProblems.LogDensityOrder{1}()
    else
        LogDensityProblems.LogDensityOrder{0}()
    end
end
function LogDensityProblems.capabilities(e::VectorEvaluator)
    return LogDensityProblems.capabilities(typeof(e))
end

function LogDensityProblems.logdensity_and_gradient(p::AbstractPrepared, x)
    return AbstractPPL.value_and_gradient(p, x)
end
function LogDensityProblems.logdensity_and_gradient(e::VectorEvaluator{V,true}, x) where {V}
    return AbstractPPL.value_and_gradient(e, x)
end

end # module
