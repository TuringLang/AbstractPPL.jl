module AbstractPPL


include("varname.jl")
include("abstractpp.jl")


# Abstract model functions
export AbstractProbabilisticProgram

# VarName
export VarName,
    inspace,
    subsumes
export @varname


end # module
