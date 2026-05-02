module AbstractPPLLogDensityProblemsExt

using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators: Prepared, VectorEvaluator
using LogDensityProblems: LogDensityProblems

# LDP integration is restricted to vector-input evaluators; `NamedTupleEvaluator`
# does not satisfy LDP's vector-input contract. Scalar output is a runtime
# contract the user must satisfy.

LogDensityProblems.logdensity(p::Prepared{<:Any,<:VectorEvaluator}, x) = p(x)
LogDensityProblems.logdensity(e::VectorEvaluator, x) = e(x)

function LogDensityProblems.dimension(p::Prepared{<:Any,<:VectorEvaluator})
    return LogDensityProblems.dimension(p.evaluator)
end
LogDensityProblems.dimension(e::VectorEvaluator) = e.dim

# Order 0 by default. AD-backend extensions (DifferentiationInterface,
# ForwardDiff, Mooncake, etc.) must overload this for their cache type to
# advertise `LogDensityOrder{1}` — without that overload,
# `logdensity_and_gradient` will hit the `value_and_gradient!!` stub and fail.
# LDP defines `capabilities(x) = capabilities(typeof(x))`, so the type method
# alone covers both call shapes.
function LogDensityProblems.capabilities(::Type{<:Prepared{<:Any,<:VectorEvaluator}})
    return LogDensityProblems.LogDensityOrder{0}()
end
function LogDensityProblems.capabilities(::Type{<:VectorEvaluator})
    return LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.logdensity_and_gradient(p::Prepared{<:Any,<:VectorEvaluator}, x)
    val, grad = AbstractPPL.value_and_gradient!!(p, x)
    return (val, copy(grad))
end

end # module
