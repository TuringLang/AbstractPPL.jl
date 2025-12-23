module AbstractPPL

# Optics
export AbstractOptic,
    Iden,
    Index,
    Property,
    ohead,
    otail,
    olast,
    oinit,
    # VarName
    VarName,
    getsym,
    getoptic,
    concretize,
    is_dynamic,
    @varname,
    @opticof,
    varname_to_optic,
    optic_to_varname

# subsumes,
# subsumedby,
# index_to_dict,
# dict_to_index,
# varname_to_string,
# string_to_varname,
# prefix,
# unprefix,
# getvalue,
# hasvalue,
# varname_leaves,
# varname_and_value_leaves

# Abstract model functions
export AbstractProbabilisticProgram,
    condition, decondition, fix, unfix, logdensityof, densityof, AbstractContext, evaluate!!

# Abstract traces
export AbstractModelTrace

include("abstractmodeltrace.jl")
include("abstractprobprog.jl")
include("evaluate.jl")
include("varname/optic.jl")
include("varname/varname.jl")
# include("varname/subsumes.jl")
# include("varname/hasvalue.jl")
# include("varname/leaves.jl")
# include("varname/prefix.jl")
# include("varname/serialize.jl")

end # module
