module AbstractPPLLogDensityProblemsExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems: AbstractPrepared, VectorEvaluator, NamedTupleEvaluator
using LogDensityProblems: LogDensityProblems

LogDensityProblems.logdensity(p::AbstractPrepared, x) = p(x)
LogDensityProblems.logdensity(e::VectorEvaluator, x) = e(x)
LogDensityProblems.logdensity(e::NamedTupleEvaluator, x::NamedTuple) = e(x)

function LogDensityProblems.dimension(p::AbstractPrepared)
    return LogDensityProblems.dimension(p.evaluator)
end
LogDensityProblems.dimension(e::VectorEvaluator) = AbstractPPL.ADProblems.dimension(e)
function LogDensityProblems.dimension(e::NamedTupleEvaluator)
    return AbstractPPL.Utils.flat_length(e.inputspec)
end

function LogDensityProblems.capabilities(p::AbstractPrepared{:gradient})
    # NamedTuple-backed evaluators can't serve the LDP vector interface;
    # logdensity_and_gradient expects a NamedTuple input, not a flat vector.
    p.evaluator isa NamedTupleEvaluator && return LogDensityProblems.LogDensityOrder{0}()
    return LogDensityProblems.LogDensityOrder{1}()
end
function LogDensityProblems.capabilities(::AbstractPrepared)
    return LogDensityProblems.LogDensityOrder{0}()
end
function LogDensityProblems.capabilities(::NamedTupleEvaluator)
    return LogDensityProblems.LogDensityOrder{0}()
end
# dim=0 evaluators implement value_and_gradient directly; treat them as gradient-capable.
function LogDensityProblems.capabilities(::VectorEvaluator{C,true}) where {C}
    return LogDensityProblems.LogDensityOrder{1}()
end

function LogDensityProblems.logdensity_and_gradient(p::AbstractPrepared{:gradient}, x)
    return AbstractPPL.value_and_gradient(p, x)
end
function LogDensityProblems.logdensity_and_gradient(e::VectorEvaluator{C,true}, x) where {C}
    return AbstractPPL.value_and_gradient(e, x)
end

end # module
