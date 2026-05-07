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

"""
    generate_testcases(::Val{group})

Return a tuple of test cases for the conformance `group`. Implemented by the
`Test` extension (`AbstractPPLTestExt`). Reserved group keys (extensions must
not redefine these): `:vector` for value/gradient/jacobian round-trips on
vector-input evaluators; `:edge` for error-path cases. Downstream packages may
add their own group keys (e.g. `:my_backend_group`) by adding methods to this
function.
"""
function generate_testcases end

"""
    run_testcases(::Val{group}, prepare_fn=AbstractPPL.prepare; adtype, kwargs...)

Run the test cases produced by [`generate_testcases`](@ref) against an AD
backend, using `prepare_fn` (default `AbstractPPL.prepare`) to construct each
prepared evaluator. Implemented by the `Test` extension. See
[`generate_testcases`](@ref) for reserved group keys.
"""
function run_testcases end

@static if VERSION >= v"1.11.0"
    eval(
        Meta.parse(
            "public prepare, value_and_gradient!!, value_and_jacobian!!, generate_testcases, run_testcases",
        ),
    )
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
