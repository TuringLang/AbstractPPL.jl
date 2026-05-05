module AbstractPPLDifferentiationInterfaceLogDensityProblemsExt

using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators: Prepared, VectorEvaluator
using LogDensityProblems: LogDensityProblems

# `DICache` lives in the DI extension, which isn't a named dependency of
# this triple extension — so resolve it via `Base.get_extension` at load
# time and register the capability method then.
function __init__()
    di_ext = Base.get_extension(AbstractPPL, :AbstractPPLDifferentiationInterfaceExt)
    DICache = di_ext.DICache
    @eval function LogDensityProblems.capabilities(
        ::Type{<:Prepared{<:Any,<:VectorEvaluator,<:$DICache}}
    )
        return LogDensityProblems.LogDensityOrder{1}()
    end
end

end # module
