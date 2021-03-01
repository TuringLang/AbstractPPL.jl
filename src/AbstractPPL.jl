module AbstractPPL


include("varname.jl")
include("abstractpp.jl")


# Abstract model functions
export AbstractProbabilisticProgram

# VarName
export inspace,
    getsym,
    getindexing,
    subsumes,
    VarName
export @varname


end # module
