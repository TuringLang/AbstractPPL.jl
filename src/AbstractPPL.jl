module AbstractPPL


include("varname.jl")
include("abstractpp.jl")
include("abstracttrace.jl")


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
