module AbstractPPL

# VarName
export VarName,
    getsym,
    getoptic,
    inspace,
    subsumes,
    subsumedby,
    varname,
    vsym,
    @varname,
    @vsym,
    index_to_dict,
    dict_to_index,
    vn_to_string,
    vn_from_string


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
