module AbstractPPL

# VarName
export VarName, getsym, getlens, inspace, subsumes, varname, vsym, @varname, @vsym


# Abstract model functions
export AbstractProbabilisticProgram, condition, decondition, logdensityof, densityof

# GraphInfo
export GraphInfo, Model, dag, nodes

# Abstract traces
export AbstractModelTrace


include("varname.jl")
include("abstractmodeltrace.jl")
include("abstractprobprog.jl")
include("deprecations.jl")
include("graphinfo.jl")
end # module
