module AbstractPPL


include("varname.jl")
include("abstractprobprog.jl")
include("abstractmodeltrace.jl")


# VarName
export inspace,
    getsym,
    getindexing,
    subsumes,
    VarName
export @varname


# Abstract model functions
export AbstractProbabilisticProgram


# Abstract traces
export AbstractModelTrace


end # module
