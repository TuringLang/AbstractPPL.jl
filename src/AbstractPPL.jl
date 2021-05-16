module AbstractPPL

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
export AbstractProbabilisticProgram,
    condition,
    decondition,
    logdensity


# Abstract traces
export AbstractModelTrace

include("varname.jl")
include("abstractprobprog.jl")
include("abstractmodeltrace.jl")
include("deprecations.jl")

end # module
