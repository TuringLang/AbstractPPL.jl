module AbstractPPLLogDensityProblemsExt

using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators: Prepared, VectorEvaluator
using ADTypes: AbstractADType
using LogDensityProblems: LogDensityProblems

LogDensityProblems.logdensity(p::Prepared{<:AbstractADType,<:VectorEvaluator}, x) = p(x)
LogDensityProblems.logdensity(e::VectorEvaluator, x) = e(x)

function LogDensityProblems.dimension(p::Prepared{<:AbstractADType,<:VectorEvaluator})
    LogDensityProblems.dimension(p.evaluator)
end
LogDensityProblems.dimension(e::VectorEvaluator) = e.dim

# AD-backend cache convention: a cache with non-`Nothing` `gradient_prep` and
# `Nothing` `jacobian_prep` denotes a scalar-output prep where
# `value_and_gradient!!` is structurally guaranteed to succeed. Caches that
# follow this convention (e.g. `DICache` from the DI extension) advertise
# `LogDensityOrder{1}` automatically; everything else stays at order 0.
function _scalar_gradient_cache(::Type{C}) where {C}
    return hasfield(C, :gradient_prep) &&
           hasfield(C, :jacobian_prep) &&
           fieldtype(C, :gradient_prep) !== Nothing &&
           fieldtype(C, :jacobian_prep) === Nothing
end

function LogDensityProblems.capabilities(
    ::Type{<:Prepared{<:AbstractADType,<:VectorEvaluator,C}}
) where {C}
    return _scalar_gradient_cache(C) ? LogDensityProblems.LogDensityOrder{1}() :
           LogDensityProblems.LogDensityOrder{0}()
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
