module AbstractPPL

# VarName
export VarName, getsym, getindexing, getlens, inspace, subsumes, varname, vsym, @varname, @vsym


# Abstract model functions
export AbstractProbabilisticProgram, condition, decondition, logdensity


# Abstract traces
export AbstractModelTrace

include("varname.jl")
include("abstractprobprog.jl")
include("abstractmodeltrace.jl")
include("deprecations.jl")

end # module
