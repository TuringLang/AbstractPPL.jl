module AbstractPPLLogDensityProblemsExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems: AbstractPrepared, VectorEvaluator, NamedTupleEvaluator
using LogDensityProblems: LogDensityProblems

LogDensityProblems.logdensity(p::AbstractPrepared, x) = p(x)
LogDensityProblems.logdensity(e::VectorEvaluator, x) = e(x)

function LogDensityProblems.dimension(p::AbstractPrepared)
    return LogDensityProblems.dimension(p.evaluator)
end
LogDensityProblems.dimension(e::VectorEvaluator) = AbstractPPL.ADProblems.dimension(e)

# Type-level capabilities (required by the LDP convention: capabilities(ℓ) = capabilities(typeof(ℓ)))
function LogDensityProblems.capabilities(::Type{<:AbstractPrepared{:gradient}})
    return LogDensityProblems.LogDensityOrder{1}()
end
function LogDensityProblems.capabilities(::Type{<:AbstractPrepared})
    return LogDensityProblems.LogDensityOrder{0}()
end

# Value-level capabilities override the type-level for NT-backed gradient evaluators,
# which cannot serve the LDP flat-vector interface.
function LogDensityProblems.capabilities(p::AbstractPrepared{:gradient})
    p.evaluator isa NamedTupleEvaluator && return LogDensityProblems.LogDensityOrder{0}()
    return LogDensityProblems.LogDensityOrder{1}()
end
function LogDensityProblems.capabilities(::AbstractPrepared)
    return LogDensityProblems.LogDensityOrder{0}()
end
# dim=0 evaluators implement value_and_gradient directly; treat them as gradient-capable.
function LogDensityProblems.capabilities(::VectorEvaluator{V,true}) where {V}
    return LogDensityProblems.LogDensityOrder{1}()
end

function LogDensityProblems.logdensity_and_gradient(p::AbstractPrepared{:gradient}, x)
    return AbstractPPL.value_and_gradient(p, x)
end
function LogDensityProblems.logdensity_and_gradient(e::VectorEvaluator{V,true}, x) where {V}
    return AbstractPPL.value_and_gradient(e, x)
end

end # module
