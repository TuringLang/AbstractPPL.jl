module AbstractPPLLogDensityProblemsExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems: AbstractPrepared, VectorEvaluator
using LogDensityProblems: LogDensityProblems

LogDensityProblems.logdensity(p::AbstractPrepared, x) = p(x)
LogDensityProblems.logdensity(e::VectorEvaluator, x) = e(x)

# Delegates to the internal implementation: works for VectorEvaluator-backed
# evaluators, throws for NamedTupleEvaluator-backed ones (no fixed vector size).
LogDensityProblems.dimension(p::AbstractPrepared) = AbstractPPL.ADProblems.dimension(p)
LogDensityProblems.dimension(e::VectorEvaluator) = AbstractPPL.ADProblems.dimension(e)

LogDensityProblems.capabilities(::AbstractPrepared{:gradient}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.capabilities(::AbstractPrepared) = LogDensityProblems.LogDensityOrder{0}()
# dim=0 evaluators implement value_and_gradient directly; treat them as gradient-capable.
LogDensityProblems.capabilities(::VectorEvaluator{C,true}) where {C} = LogDensityProblems.LogDensityOrder{1}()

LogDensityProblems.logdensity_and_gradient(p::AbstractPrepared{:gradient}, x) =
    AbstractPPL.value_and_gradient(p, x)
LogDensityProblems.logdensity_and_gradient(e::VectorEvaluator{C,true}, x) where {C} =
    AbstractPPL.value_and_gradient(e, x)

end # module
