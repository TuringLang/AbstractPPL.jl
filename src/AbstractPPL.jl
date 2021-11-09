module AbstractPPL

# VarName
export VarName, getsym, getindexing, getlens, inspace, subsumes, varname, vsym, @varname, @vsym


# Abstract model functions
export AbstractProbabilisticProgram, condition, decondition, logdensityof, densityof


# Abstract traces
export AbstractModelTrace


include("varname.jl")
include("abstractmodeltrace.jl")
include("abstractprobprog.jl")
include("deprecations.jl")

end # module
