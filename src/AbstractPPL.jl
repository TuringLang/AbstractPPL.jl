module AbstractPPL

# VarName
export VarName,
    getsym,
    getoptic,
    inspace,
    subsumes,
    subsumedby,
    varname,
    vn_to_string,
    vn_from_string,
    vsym,
    @varname,
    @vsym


# Abstract model functions
export AbstractProbabilisticProgram, condition, decondition, logdensityof, densityof, AbstractContext, evaluate!!

# Abstract traces
export AbstractModelTrace


include("varname.jl")
include("abstractmodeltrace.jl")
include("abstractprobprog.jl")
include("evaluate.jl")
include("deprecations.jl")

end # module
