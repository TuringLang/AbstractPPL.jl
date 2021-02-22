module AbstractPPL


include("varname.jl")
include("contexts.jl")
include("abstractpp.jl")


# Abstract model functions
export AbstractProbabilisticProgram
export generate,
    logdensity,
    logjoint,
    loglikelihood,
    logprior,
    sample

# VarName
export VarName,
    inspace,
    subsumes
export @varname


end # module
