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

# Type-level capabilities (required by the LDP convention: capabilities(ℓ) = capabilities(typeof(ℓ)))
function LogDensityProblems.capabilities(::Type{<:AbstractPrepared})
    return LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.capabilities(p::AbstractPrepared)
    return if AbstractPPL.ADProblems._supports_gradient(p)
        LogDensityProblems.LogDensityOrder{1}()
    else
        LogDensityProblems.LogDensityOrder{0}()
    end
end

function LogDensityProblems.capabilities(::VectorEvaluator{V,true}) where {V}
    return LogDensityProblems.LogDensityOrder{1}()
end

function LogDensityProblems.logdensity_and_gradient(p::AbstractPrepared, x)
    return AbstractPPL.value_and_gradient(p, x)
end
function LogDensityProblems.logdensity_and_gradient(e::VectorEvaluator{V,true}, x) where {V}
    return AbstractPPL.value_and_gradient(e, x)
end

end # module
