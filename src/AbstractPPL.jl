module AbstractPPL


include("varname.jl")
include("contexts.jl")
include("abstractprobprog.jl")
include("abstractmodeltrace.jl")


# VarName
export VarName,
    getsym,
    getindexing,
    inspace,
    subsumes,
    varname,
    vinds,
    vsym,
    @varname,
    @vinds,
    @vsym


# Abstract model functions
export AbstractProbabilisticProgram
export generate,
    logdensity,
    logjoint,
    loglikelihood,
    logprior,
    sample


# Abstract traces
export AbstractModelTrace


end # module
