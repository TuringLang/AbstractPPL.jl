module AbstractPPLTestExt

using AbstractPPL: AbstractPPL, generate_testcases, run_testcase
using Test: @inferred, @test, @test_broken, @test_throws, @testset

"""
    TestCase(name, tag, f, x_proto; x, value, gradient, jacobian, hessian,
             context=(), op, exception, inputs, allocations_safe=true)

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

`allocations_safe=false` opts the case out of the alloc check
(cases with an allocating primal or empty-input shortcuts that allocate).
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

struct QuadraticProblem end
(::QuadraticProblem)(x::AbstractVector{<:Real}) = sum(xi -> xi^2, x)

struct VectorValuedProblem end
(::VectorValuedProblem)(x::AbstractVector{<:Real}) = [x[1] * x[2], x[2] + x[3]]

_context_problem(y::AbstractVector{<:Real}, offset) = -0.5 * (y[1] - offset)^2

function AbstractPPL.generate_testcases(::Val{:vector})
    return (
        TestCase(
            "quadratic (scalar output)",
            :vector,
            QuadraticProblem(),
            zeros(3);
            x=[3.0, 1.0, 2.0],
            value=14.0,
            gradient=[6.0, 2.0, 4.0],
        ),
        TestCase(
            "vector-valued (vector output)",
            :vector,
            VectorValuedProblem(),
            zeros(3);
            x=[2.0, 3.0, 4.0],
            value=[6.0, 7.0],
            jacobian=[3.0 2.0 0.0; 0.0 1.0 1.0],
            allocations_safe=false,  # primal allocates its result vector
        ),
        TestCase(
            "empty input, scalar output",
            :vector,
            x -> 7.5,
            Float64[];
            x=Float64[],
            value=7.5,
            gradient=Float64[],
            allocations_safe=false,  # empty-input shortcut returns fresh `T[]`
        ),
        TestCase(
            "empty input, vector output",
            :vector,
            x -> [2.0, 3.0],
            Float64[];
            x=Float64[],
            value=[2.0, 3.0],
            jacobian=zeros(2, 0),
            allocations_safe=false,  # empty-input shortcut allocates empty matrix
        ),
        TestCase(
            "scalar gradient with context",
            :context,
            _context_problem,
            [0.3];
            x=[0.3],
            value=_context_problem([0.3], 0.1),
            gradient=[-(0.3 - 0.1)],
            context=(0.1,),
        ),
        TestCase(
            "quadratic (hessian)",
            :hessian,
            QuadraticProblem(),
            zeros(3);
            x=[3.0, 1.0, 2.0],
            value=14.0,
            gradient=[6.0, 2.0, 4.0],
            hessian=[2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0],
            allocations_safe=false,  # ForwardDiff/Mooncake hessian path allocates scratch
        ),
        TestCase(
            "empty input, hessian",
            :hessian,
            x -> 7.5,
            Float64[];
            x=Float64[],
            value=7.5,
            gradient=Float64[],
            hessian=zeros(0, 0),
            allocations_safe=false,
        ),
        # value_gradient_and_hessian!! rejects order=1 preps regardless of arity;
        # both paths share the dispatch so one case suffices.
        TestCase(
            "value_gradient_and_hessian!! on order=1 prep",
            :edge,
            QuadraticProblem(),
            zeros(3);
            x=[3.0, 1.0, 2.0],
            op=(prepared, x) -> AbstractPPL.value_gradient_and_hessian!!(prepared, x),
            exception=r"order=2",
        ),
        TestCase(
            "wrong vector length",
            :edge,
            QuadraticProblem(),
            zeros(3);
            x=[3.0, 1.0, 2.0, 99.0],
            op=(prepared, x) -> prepared(x),
            exception=DimensionMismatch,
        ),
        TestCase(
            "non-floating-point vector",
            :edge,
            QuadraticProblem(),
            zeros(3);
            x=[3, 1, 2],
            op=(prepared, x) -> prepared(x),
            exception=r"floating-point",
        ),
        TestCase(
            "gradient of vector-valued output",
            :edge,
            VectorValuedProblem(),
            zeros(3);
            x=[2.0, 3.0, 4.0],
            op=(prepared, x) -> AbstractPPL.value_and_gradient!!(prepared, x),
            exception=r"scalar-valued",
        ),
        TestCase(
            "jacobian of scalar output",
            :edge,
            QuadraticProblem(),
            zeros(3);
            x=[3.0, 1.0, 2.0],
            op=(prepared, x) -> AbstractPPL.value_and_jacobian!!(prepared, x),
            exception=r"vector-valued",
        ),
        TestCase(
            "gradient of vector-valued output, empty input",
            :edge,
            x -> [2.0, 3.0],
            Float64[];
            x=Float64[],
            op=(prepared, x) -> AbstractPPL.value_and_gradient!!(prepared, x),
            exception=r"scalar-valued",
        ),
        TestCase(
            "jacobian of scalar output, empty input",
            :edge,
            x -> 7.5,
            Float64[];
            x=Float64[],
            op=(prepared, x) -> AbstractPPL.value_and_jacobian!!(prepared, x),
            exception=r"vector-valued",
        ),
        TestCase(
            "value_and_gradient!! wrong vector length",
            :edge,
            QuadraticProblem(),
            zeros(3);
            x=[3.0, 1.0, 2.0, 99.0],
            op=(prepared, x) -> AbstractPPL.value_and_gradient!!(prepared, x),
            exception=DimensionMismatch,
        ),
        TestCase(
            "value_and_jacobian!! wrong vector length",
            :edge,
            VectorValuedProblem(),
            zeros(3);
            x=[2.0, 3.0, 4.0, 5.0],
            op=(prepared, x) -> AbstractPPL.value_and_jacobian!!(prepared, x),
            exception=DimensionMismatch,
        ),
        TestCase(
            "value_and_gradient!! non-floating-point vector",
            :edge,
            QuadraticProblem(),
            zeros(3);
            x=[3, 1, 2],
            op=(prepared, x) -> AbstractPPL.value_and_gradient!!(prepared, x),
            exception=r"floating-point",
        ),
        TestCase(
            "value_and_jacobian!! non-floating-point vector",
            :edge,
            VectorValuedProblem(),
            zeros(3);
            x=[2, 3, 4],
            op=(prepared, x) -> AbstractPPL.value_and_jacobian!!(prepared, x),
            exception=r"floating-point",
        ),
        TestCase(
            "scalar output, cache reuse",
            :cache_reuse,
            QuadraticProblem(),
            zeros(3);
            inputs=[
                (x=[1.0, 2.0, 3.0], value=14.0, gradient=[2.0, 4.0, 6.0]),
                (x=[4.0, 5.0, 6.0], value=77.0, gradient=[8.0, 10.0, 12.0]),
                (x=[0.5, -1.0, 2.0], value=5.25, gradient=[1.0, -2.0, 4.0]),
            ],
            allocations_safe=false,
        ),
        TestCase(
            "vector output, cache reuse",
            :cache_reuse,
            VectorValuedProblem(),
            zeros(3);
            inputs=[
                (x=[2.0, 3.0, 4.0], value=[6.0, 7.0], jacobian=[3.0 2.0 0.0; 0.0 1.0 1.0]),
                (x=[5.0, 1.0, 7.0], value=[5.0, 8.0], jacobian=[1.0 5.0 0.0; 0.0 1.0 1.0]),
                (x=[0.0, 4.0, -2.0], value=[0.0, 2.0], jacobian=[4.0 0.0 0.0; 0.0 1.0 1.0]),
            ],
            allocations_safe=false,
        ),
    )
end

function AbstractPPL.generate_testcases(::Val{:namedtuple})
    return (
        TestCase(
            "scalar output over (x::Real, y::Vector)",
            :namedtuple,
            vs -> vs.x^2 + sum(abs2, vs.y),
            (x=0.0, y=zeros(2));
            x=(x=3.0, y=[1.0, 2.0]),
            value=14.0,
            gradient=(x=6.0, y=[2.0, 4.0]),
        ),
    )
end

# ----- helpers -----

# NamedTuple gradients compare per-key (some backends return Mooncake-tagged
# tangents that aren't directly `≈`-comparable as a whole).
function _compare_derivative(actual::NamedTuple, expected::NamedTuple; atol, rtol)
    for k in keys(expected)
        @test getproperty(actual, k) ≈ getproperty(expected, k) atol = atol rtol = rtol
    end
end
function _compare_derivative(actual, expected; atol, rtol)
    @test actual ≈ expected atol = atol rtol = rtol
end

function _record_alloc!(state::Symbol, allocs::Integer)
    state === :test && @test allocs == 0
    state === :broken && @test_broken allocs == 0
    return nothing
end

# `@inferred` is syntactic and throws on failure; wrap so we can pin `op`'s
# type via an F-parameter and convert the throw into a Bool.
function _is_inferred(op::F, args...) where {F}
    try
        @inferred op(args...)
        return true
    catch
        return false
    end
end

function _record_inferred!(state::Symbol, inferred::Bool)
    state === :test && @test inferred
    state === :broken && @test_broken inferred
    return nothing
end

# ----- runner -----

function AbstractPPL.run_testcase(case::TestCase; kwargs...)
    @testset "$(case.name)" begin
        _run(Val(case.tag), case; kwargs...)
    end
    return nothing
end

function _run(
    ::Val{:vector},
    case;
    adtype,
    prepare_fn=AbstractPPL.prepare,
    atol=0,
    rtol=1e-10,
    check_dims::Bool=true,
    type_stability::Symbol=:skip,
    allocations::Symbol=:skip,
)
    prepared = prepare_fn(adtype, case.f, case.x_proto; check_dims)
    @test AbstractPPL.order(prepared) == 1
    @test prepared(case.x) ≈ case.value atol = atol rtol = rtol

    if case.gradient !== nothing
        val, grad = AbstractPPL.value_and_gradient!!(prepared, case.x)
        @test val ≈ case.value atol = atol rtol = rtol
        _compare_derivative(grad, case.gradient; atol, rtol)
        _maybe_check_alloc!(
            case, allocations, AbstractPPL.value_and_gradient!!, prepared, case.x
        )
        _maybe_check_inferred!(
            type_stability, AbstractPPL.value_and_gradient!!, prepared, case.x
        )
    end

    if case.jacobian !== nothing
        val, jac = AbstractPPL.value_and_jacobian!!(prepared, case.x)
        @test val ≈ case.value atol = atol rtol = rtol
        @test jac ≈ case.jacobian atol = atol rtol = rtol
        _maybe_check_alloc!(
            case, allocations, AbstractPPL.value_and_jacobian!!, prepared, case.x
        )
        _maybe_check_inferred!(
            type_stability, AbstractPPL.value_and_jacobian!!, prepared, case.x
        )
    end
    return nothing
end

function _run(
    ::Val{:context},
    case;
    adtype,
    prepare_fn=AbstractPPL.prepare,
    atol=0,
    rtol=1e-10,
    check_dims::Bool=true,
    type_stability::Symbol=:skip,
    allocations::Symbol=:skip,
)
    prepared = prepare_fn(adtype, case.f, case.x_proto; check_dims, context=case.context)
    @test AbstractPPL.order(prepared) == 1
    @test prepared(case.x) ≈ case.value atol = atol rtol = rtol
    val, grad = AbstractPPL.value_and_gradient!!(prepared, case.x)
    @test val ≈ case.value atol = atol rtol = rtol
    @test grad ≈ case.gradient atol = atol rtol = rtol
    _maybe_check_alloc!(
        case, allocations, AbstractPPL.value_and_gradient!!, prepared, case.x
    )
    _maybe_check_inferred!(
        type_stability, AbstractPPL.value_and_gradient!!, prepared, case.x
    )
    return nothing
end

function _run(
    ::Val{:hessian},
    case;
    adtype,
    prepare_fn=AbstractPPL.prepare,
    atol=0,
    rtol=1e-10,
    check_dims::Bool=true,
    type_stability::Symbol=:skip,
    allocations::Symbol=:skip,
)
    prepared = prepare_fn(adtype, case.f, case.x_proto; check_dims, order=2)
    @test AbstractPPL.order(prepared) == 2
    @test prepared(case.x) ≈ case.value atol = atol rtol = rtol

    val, grad, hess = AbstractPPL.value_gradient_and_hessian!!(prepared, case.x)
    @test val ≈ case.value atol = atol rtol = rtol
    @test grad ≈ case.gradient atol = atol rtol = rtol
    @test hess ≈ case.hessian atol = atol rtol = rtol

    # Order=2 prep also satisfies the order=1 gradient contract.
    val1, grad1 = AbstractPPL.value_and_gradient!!(prepared, case.x)
    @test val1 ≈ case.value atol = atol rtol = rtol
    @test grad1 ≈ case.gradient atol = atol rtol = rtol

    _maybe_check_alloc!(
        case, allocations, AbstractPPL.value_gradient_and_hessian!!, prepared, case.x
    )
    _maybe_check_inferred!(
        type_stability, AbstractPPL.value_gradient_and_hessian!!, prepared, case.x
    )
    return nothing
end

function _run(::Val{:edge}, case; adtype, prepare_fn=AbstractPPL.prepare, kwargs...)
    prepared = prepare_fn(adtype, case.f, case.x_proto)
    @test_throws case.exception case.op(prepared, case.x)
    return nothing
end

function _run(
    ::Val{:cache_reuse},
    case;
    adtype,
    prepare_fn=AbstractPPL.prepare,
    atol=0,
    rtol=1e-10,
    kwargs...,
)
    prepared = prepare_fn(adtype, case.f, case.x_proto)
    for input in case.inputs
        if haskey(input, :gradient)
            val, grad = AbstractPPL.value_and_gradient!!(prepared, input.x)
            @test val ≈ input.value atol = atol rtol = rtol
            @test grad ≈ input.gradient atol = atol rtol = rtol
        else
            val, jac = AbstractPPL.value_and_jacobian!!(prepared, input.x)
            @test val ≈ input.value atol = atol rtol = rtol
            @test jac ≈ input.jacobian atol = atol rtol = rtol
        end
    end
    return nothing
end

function _run(
    ::Val{:namedtuple},
    case;
    adtype,
    prepare_fn=AbstractPPL.prepare,
    atol=0,
    rtol=1e-10,
    kwargs...,
)
    prepared = prepare_fn(adtype, case.f, case.x_proto)
    @test prepared(case.x) ≈ case.value atol = atol rtol = rtol
    val, grad = AbstractPPL.value_and_gradient!!(prepared, case.x)
    @test val ≈ case.value atol = atol rtol = rtol
    _compare_derivative(grad, case.gradient; atol, rtol)
    return nothing
end

_resolve_alloc_state(case::TestCase, state::Symbol) = case.allocations_safe ? state : :skip

function _maybe_check_alloc!(case::TestCase, state::Symbol, op::F, prepared, x) where {F}
    effective = _resolve_alloc_state(case, state)
    effective === :skip && return nothing
    op(prepared, x)  # warm up
    allocs = @allocated op(prepared, x)
    return _record_alloc!(effective, allocs)
end

function _maybe_check_inferred!(state::Symbol, op::F, prepared, x) where {F}
    state === :skip && return nothing
    return _record_inferred!(state, _is_inferred(op, prepared, x))
end

end # module
