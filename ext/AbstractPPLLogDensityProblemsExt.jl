module AbstractPPLLogDensityProblemsExt

using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators: Prepared, VectorEvaluator
using ADTypes: AbstractADType
using LogDensityProblems: LogDensityProblems

# LDP integration is restricted to vector-input evaluators; `NamedTupleEvaluator`
# does not satisfy LDP's vector-input contract. Scalar output is a runtime
# contract the user must satisfy.

LogDensityProblems.logdensity(p::Prepared{<:AbstractADType,<:VectorEvaluator}, x) = p(x)
LogDensityProblems.logdensity(e::VectorEvaluator, x) = e(x)

function LogDensityProblems.dimension(p::Prepared{<:AbstractADType,<:VectorEvaluator})
    return LogDensityProblems.dimension(p.evaluator)
end
LogDensityProblems.dimension(e::VectorEvaluator) = e.dim

# A `Prepared` (i.e. `prepare(adtype, ...)`) advertises gradient capability;
# a bare evaluator (i.e. `prepare(problem, x)`, no adtype) is primal-only.
# Backends that don't implement `value_and_gradient!!` for their `Prepared`
# type will surface a `MethodError` at call time — that's the trade-off of
# this rule, in exchange for a uniform contract.
function LogDensityProblems.capabilities(
    ::Type{<:Prepared{<:AbstractADType,<:VectorEvaluator}}
)
    return LogDensityProblems.LogDensityOrder{1}()
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
