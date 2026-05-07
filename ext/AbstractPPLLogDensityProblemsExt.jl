module AbstractPPLLogDensityProblemsExt

using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators: Prepared, VectorEvaluator
using ADTypes: AbstractADType
using LogDensityProblems: LogDensityProblems

# LDP integration is restricted to vector-input evaluators; `NamedTupleEvaluator`
# does not satisfy LDP's vector-input contract. Scalar-output is a separate
# capability advertised by AD-backend cross-extensions (e.g. the DI × LDP
# extension overloads `capabilities` for `DICache` shapes with scalar output).

LogDensityProblems.logdensity(p::Prepared{<:AbstractADType,<:VectorEvaluator}, x) = p(x)
LogDensityProblems.logdensity(e::VectorEvaluator, x) = e(x)

function LogDensityProblems.dimension(p::Prepared{<:AbstractADType,<:VectorEvaluator})
    return LogDensityProblems.dimension(p.evaluator)
end
LogDensityProblems.dimension(e::VectorEvaluator) = e.dim

# Order 0 by default. Backend-specific cross-extensions opt into
# `LogDensityOrder{1}` on their concrete cache type when the cache shape proves
# `value_and_gradient!!` will succeed (scalar output, gradient prep populated).
function LogDensityProblems.capabilities(
    ::Type{<:Prepared{<:AbstractADType,<:VectorEvaluator}}
)
    return LogDensityProblems.LogDensityOrder{0}()
end
function LogDensityProblems.capabilities(::Type{<:VectorEvaluator})
    return LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.logdensity_and_gradient(
    p::Prepared{<:AbstractADType,<:VectorEvaluator}, x
)
    val, grad = AbstractPPL.value_and_gradient!!(p, x)
    # `value_and_gradient!!` may alias internal storage; LDP requires a stable result.
    return (val, copy(grad))
end

end # module
