module AbstractPPL

# Abstract model functions
export AbstractProbabilisticProgram,
    condition, decondition, fix, unfix, logdensityof, densityof, AbstractContext, evaluate!!

# Abstract traces
export AbstractModelTrace

include("abstractmodeltrace.jl")
include("abstractprobprog.jl")
include("evaluate.jl")
include("evaluators/Evaluators.jl")
using .Evaluators: prepare, value_and_gradient!!, value_and_jacobian!!
@static if VERSION >= v"1.11.0"
    eval(Meta.parse("public prepare, value_and_gradient!!, value_and_jacobian!!"))
end

include("varname/optic.jl")
include("varname/varname.jl")
include("varname/subsumes.jl")
include("varname/hasvalue.jl")
include("varname/leaves.jl")
include("varname/prefix.jl")
include("varname/serialize.jl")

# Optics
export AbstractOptic,
    Iden,
    Index,
    Property,
    with_mutation,
    ohead,
    otail,
    olast,
    oinit,
    # VarName
    VarName,
    getsym,
    getoptic,
    concretize,
    concretize_top_level,
    is_dynamic,
    @varname,
    varname,
    @opticof,
    varname_to_optic,
    optic_to_varname,
    append_optic,
    # other functions
    subsumes,
    prefix,
    unprefix,
    hasvalue,
    getvalue,
    canview,
    varname_leaves,
    varname_and_value_leaves,
    # Serialisation
    index_to_dict,
    dict_to_index,
    varname_to_string,
    string_to_varname

# Convenience re-export
using Accessors: set
export set

end # module
