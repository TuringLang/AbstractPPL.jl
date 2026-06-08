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
using .Evaluators:
    prepare, value_and_gradient!!, value_and_jacobian!!, value_gradient_and_hessian!!, order

"""
    generate_testcases(::Val{group})

Return a tuple of AD conformance test cases for the input-shape `group`.
Reserved groups: `:vector` (vector input) and `:namedtuple` (NamedTuple
input; Mooncake-only). Iterate and pass each to [`run_testcase`](@ref).
Implemented by the `Test` extension (`AbstractPPLTestExt`).
"""
function generate_testcases end

"""
    run_testcase(case; adtype, prepare_fn=AbstractPPL.prepare, atol=0, rtol=1e-10,
                 check_dims=true, type_stability=:skip, allocations=:skip)

Run a single conformance case against an AD backend. `type_stability` and
`allocations` accept `:skip` / `:test` / `:broken` — `:test` asserts the
invariant, `:broken` marks it `@test_broken` (use for backends with known
regressions). Implemented by the `Test` extension.
"""
function run_testcase end

@static if VERSION >= v"1.11.0"
    eval(
        Meta.parse(
            "public prepare, value_and_gradient!!, value_and_jacobian!!, " *
            "value_gradient_and_hessian!!, order, " *
            "generate_testcases, run_testcase",
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

include("of.jl")
export of, @of
@static if VERSION >= v"1.11.0"
    eval(
        Meta.parse(
            "public OfType, OfReal, OfInt, OfArray, OfNamedTuple, OfConstantWrapper, " *
            "flatten, unflatten",
        ),
    )
end

end # module
