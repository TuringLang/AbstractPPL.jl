module AbstractPPL


include("varname.jl")
include("abstractmodel.jl")
include("abstractvarinfo.jl")


# Abstract model functions
export AbstractModel
export logjoint,
    logprior

# VarName
export VarName,
    inspace,
    subsumes
export @varname

# VarInfo
export AbstractVarInfo


end # module
