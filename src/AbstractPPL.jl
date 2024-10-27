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
    varname_to_string,
    string_to_varname


# Abstract model functions
export AbstractProbabilisticProgram, condition, decondition, fix, unfix, logdensityof, densityof, AbstractContext, evaluate!!

# Abstract traces
export AbstractModelTrace


include("varname.jl")
include("abstractmodeltrace.jl")
include("abstractprobprog.jl")
include("evaluate.jl")
include("deprecations.jl")

end # module
