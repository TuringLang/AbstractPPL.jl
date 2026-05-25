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
    TestCase(name, tag, f, x_proto; x, value, gradient, jacobian, hessian,
             context=(), op, exception, inputs)

Single tagged case for AD conformance testing. The `tag::Symbol` selects how
the case is run; the kwargs populate only the fields the tag uses.

Reserved tags (recognised by [`run_testcase`](@ref)):

  - `:vector`      — vector input, scalar output (`gradient`) or vector output
                     (`jacobian`).
  - `:hessian`     — order=2 round-trip on scalar output.
  - `:context`     — scalar-output gradient with a non-empty `context::Tuple`
                     passed to `prepare`.
  - `:edge`        — error case; `op(prepared, x)` must throw `exception`.
  - `:cache_reuse` — multiple inputs against a single prepared evaluator
                     (`inputs::Vector{<:NamedTuple}`, with `(x=, value=,
                     gradient=)` or `(x=, value=, jacobian=)` per row).
  - `:namedtuple`  — NamedTuple input and gradient; Mooncake-only.
"""
struct TestCase
    name::String
    tag::Symbol
    f::Any
    x_proto::Any
    x::Any
    value::Any
    gradient::Any
    jacobian::Any
    hessian::Any
    context::Tuple
    op::Any
    exception::Any
    inputs::Any
    # Cases with an allocating primal (vector-output result vectors, the
    # empty-input shortcut's `T[]`) or shapes the original `:allocations` group
    # never covered (hessian, cache-reuse, edge) set this to `false` — the
    # runner then skips the `allocations=` check regardless of caller intent.
    allocations_safe::Bool
end
function TestCase(
    name,
    tag::Symbol,
    f,
    x_proto;
    x=nothing,
    value=nothing,
    gradient=nothing,
    jacobian=nothing,
    hessian=nothing,
    context::Tuple=(),
    op=nothing,
    exception=nothing,
    inputs=nothing,
    allocations_safe::Bool=true,
)
    return TestCase(
        name,
        tag,
        f,
        x_proto,
        x,
        value,
        gradient,
        jacobian,
        hessian,
        context,
        op,
        exception,
        inputs,
        allocations_safe,
    )
end

"""
    generate_testcases()

Return a tuple of conformance [`TestCase`](@ref)s for vector-input AD
backends. Iterate and pass each to [`run_testcase`](@ref).
"""
function generate_testcases end

"""
    generate_namedtuple_testcases()

Like [`generate_testcases`](@ref) but for evaluators with `NamedTuple` input.
"""
function generate_namedtuple_testcases end

"""
    run_testcase(case; adtype, prepare_fn=AbstractPPL.prepare, atol=0, rtol=1e-10,
                 check_dims=true, type_stability=:skip, allocations=:skip)

Run a single [`TestCase`](@ref) against an AD backend. `type_stability` and
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
            "generate_testcases, generate_namedtuple_testcases, run_testcase, TestCase",
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
