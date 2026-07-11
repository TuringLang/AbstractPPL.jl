module AbstractPPLTestExt

using AbstractPPL: AbstractPPL, generate_testcases, run_testcase
using Test: @inferred, @test, @test_broken, @test_throws, @testset

"""
    TestCase(name, tag, f, x_proto; x, value, gradient, jacobian, hessian,
             context=(), op, exception, inputs, override, allocations_safe=true)

Single tagged case for AD conformance testing. The `tag::Symbol` selects how
the case is run; the kwargs populate only the fields the tag uses.

Reserved tags (recognised by [`run_testcase`](@ref)):

  - `:vector`      — vector input, scalar output (`gradient`) or vector output
                     (`jacobian`).
  - `:hessian`     — order=2 round-trip on scalar output.
  - `:context`     — scalar-output gradient with a non-empty `context::Tuple`
                     passed to `prepare`.
  - `:context_override` — the frozen `context` (`gradient`/`hessian` expected)
                     against a per-call `override::NamedTuple` `(context, value,
                     gradient, hessian)`. The `gradient_override` /
                     `hessian_override` [`run_testcase`](@ref) kwargs select
                     whether each override is honoured or rejected per backend.
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
    override::Any
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
    override=nothing,
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
        override,
        allocations_safe,
    )
end

struct QuadraticProblem end
(::QuadraticProblem)(x::AbstractVector{<:Real}) = sum(xi -> xi^2, x)

struct VectorValuedProblem end
(::VectorValuedProblem)(x::AbstractVector{<:Real}) = [x[1] * x[2], x[2] + x[3]]

_context_problem(y::AbstractVector{<:Real}, offset) = -0.5 * (y[1] - offset)^2

# `∂/∂y a·‖y‖² = 2a·y` and its Hessian `2a·I` both depend on the context `a`, so
# a context override is directly observable in the gradient and Hessian.
_affine(y::AbstractVector{<:Real}, a, b) = a * sum(abs2, y) + b

# Array-valued context: gradient `2w²ᵢyᵢ` and Hessian `Diagonal(2w²)` reflect an
# override of the whole data vector. The empty branch only runs on the
# empty-input shortcuts, which bypass AD entirely.
_weighted(y::AbstractVector{<:Real}, w) = isempty(y) ? zero(eltype(w)) : sum(abs2, y .* w)

# Vector-output problem for the Jacobian override path: scales the input by the
# first context element, so the Jacobian is `c₁·I` (or `Diagonal(c₁)` for an
# array `c₁`) and an override is directly observable.
_jac_ctx_problem(y::AbstractVector{<:Real}, c1, rest...) = c1 .* y
function _jac_ctx_expected(context, x)
    c1 = first(context)
    n = length(x)
    J = zeros(n, n)
    for i in 1:n
        J[i, i] = c1 isa AbstractArray ? c1[i] : c1
    end
    return J
end

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
            allocations_safe=false,  # empty-input hessian shortcut allocates
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
            allocations_safe=false,  # cache-reuse loops aren't single-call alloc tests
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
            allocations_safe=false,  # cache-reuse loops aren't single-call alloc tests
        ),
    )
end

function AbstractPPL.generate_testcases(::Val{:context_override})
    x = [1.0, 2.0, 3.0]
    return (
        TestCase(
            "affine scalar output, context override",
            :context_override,
            _affine,
            zeros(3);
            x=x,
            context=(2.0, 1.0),
            value=_affine(x, 2.0, 1.0),
            gradient=4.0 .* x,   # 2a·y with a=2
            hessian=[4.0 0 0; 0 4.0 0; 0 0 4.0],
            override=(
                context=(3.0, 5.0),
                value=_affine(x, 3.0, 5.0),
                gradient=6.0 .* x,   # 2a·y with a=3
                hessian=[6.0 0 0; 0 6.0 0; 0 0 6.0],
            ),
        ),
        TestCase(
            "weighted scalar output, array-context override",
            :context_override,
            _weighted,
            zeros(3);
            x=x,
            context=([1.0, 2.0, 3.0],),
            value=_weighted(x, [1.0, 2.0, 3.0]),
            gradient=2.0 .* [1.0, 2.0, 3.0] .^ 2 .* x,
            hessian=[2.0 0 0; 0 8.0 0; 0 0 18.0],
            override=(
                context=([2.0, 1.0, 0.5],),
                value=_weighted(x, [2.0, 1.0, 0.5]),
                gradient=2.0 .* [2.0, 1.0, 0.5] .^ 2 .* x,
                hessian=[8.0 0 0; 0 2.0 0; 0 0 0.5],
            ),
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

# `:vector` and `:context` share a runner — `case.context` defaults to `()` so
# threading it through `prepare` is a no-op on `:vector` cases that don't set
# it.
function _run(
    ::Union{Val{:vector},Val{:context}},
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

# Context frozen at `prepare` vs a per-call `context=` override, on the
# gradient, Jacobian, and Hessian entry points, plus the invariants that hold on
# every backend. `gradient_override`/`hessian_override` are `:honor` (the value
# is swapped and observed) or `:reject` (the backend bakes context into its
# prepared state and throws for a non-empty input); `jacobian_override` adds
# `:prepare_rejects` for backends that refuse vector arity + non-empty context
# at `prepare` (Mooncake), where only the trivially-matching empty override
# exists. Each backend driver passes the flags that match its caches. Contract
# violations (type/arity mismatch, non-`Tuple`) are validated centrally and
# throw on every backend regardless of the flags. Empty input runs no
# derivative machinery, so no backend rejects an override there — but the
# override is still validated, and integer-eltype inputs are still rejected.
function _run(
    ::Val{:context_override},
    case;
    adtype,
    prepare_fn=AbstractPPL.prepare,
    atol=0,
    rtol=1e-10,
    check_dims::Bool=true,
    gradient_override::Symbol=:honor,
    hessian_override::Symbol=:honor,
    jacobian_override::Symbol=:honor,
    kwargs...,
)
    ov = case.override

    # --- order=1 gradient ---
    prep = prepare_fn(adtype, case.f, case.x_proto; check_dims, context=case.context)
    val, grad = AbstractPPL.value_and_gradient!!(prep, case.x)
    @test val ≈ case.value atol = atol rtol = rtol
    @test grad ≈ case.gradient atol = atol rtol = rtol
    if gradient_override === :reject
        @test_throws ArgumentError AbstractPPL.value_and_gradient!!(
            prep, case.x; context=ov.context
        )
    else
        val_o, grad_o = AbstractPPL.value_and_gradient!!(prep, case.x; context=ov.context)
        @test val_o ≈ ov.value atol = atol rtol = rtol
        @test grad_o ≈ ov.gradient atol = atol rtol = rtol
        # Matches a prep built with the override context from the start.
        fresh = prepare_fn(adtype, case.f, case.x_proto; check_dims, context=ov.context)
        @test grad_o ≈ AbstractPPL.value_and_gradient!!(fresh, case.x)[2] atol = atol rtol =
            rtol
        # The override is per-call: the next default call still uses the frozen context.
        @test AbstractPPL.value_and_gradient!!(prep, case.x)[2] ≈ case.gradient atol = atol rtol =
            rtol
    end

    # --- order=2 gradient+Hessian ---
    preph = prepare_fn(
        adtype, case.f, case.x_proto; check_dims, context=case.context, order=2
    )
    @test AbstractPPL.value_gradient_and_hessian!!(preph, case.x)[3] ≈ case.hessian atol =
        atol rtol = rtol

    # `value_and_gradient!!` on an order=2 prep is a distinct dispatch (the
    # gradient cache, not the Hessian one) that also takes the override; it
    # follows `gradient_override`, not `hessian_override`.
    if gradient_override === :reject
        @test_throws ArgumentError AbstractPPL.value_and_gradient!!(
            preph, case.x; context=ov.context
        )
    else
        @test AbstractPPL.value_and_gradient!!(preph, case.x; context=ov.context)[2] ≈
            ov.gradient atol = atol rtol = rtol
    end

    if hessian_override === :reject
        @test_throws ArgumentError AbstractPPL.value_gradient_and_hessian!!(
            preph, case.x; context=ov.context
        )
    else
        _, grad_h, hess_h = AbstractPPL.value_gradient_and_hessian!!(
            preph, case.x; context=ov.context
        )
        @test grad_h ≈ ov.gradient atol = atol rtol = rtol
        @test hess_h ≈ ov.hessian atol = atol rtol = rtol
        # Matches a fresh order=2 prep built with the override context.
        freshh = prepare_fn(
            adtype, case.f, case.x_proto; check_dims, context=ov.context, order=2
        )
        @test hess_h ≈ AbstractPPL.value_gradient_and_hessian!!(freshh, case.x)[3] atol =
            atol rtol = rtol
        # Per-call: a subsequent default call still uses the frozen context.
        @test AbstractPPL.value_gradient_and_hessian!!(preph, case.x)[3] ≈ case.hessian atol =
            atol rtol = rtol
    end

    # --- Jacobian override ---
    if jacobian_override === :prepare_rejects
        # Vector arity + non-empty context is refused at `prepare`, so a
        # context-carrying Jacobian prep cannot exist; on a context-free vector
        # prep only the trivially-matching empty override validates.
        @test_throws ArgumentError prepare_fn(
            adtype, _jac_ctx_problem, case.x_proto; check_dims, context=case.context
        )
        jprep0 = prepare_fn(adtype, VectorValuedProblem(), case.x_proto; check_dims)
        @test AbstractPPL.value_and_jacobian!!(jprep0, case.x; context=())[1] ≈
            VectorValuedProblem()(case.x) atol = atol rtol = rtol
        @test_throws ArgumentError AbstractPPL.value_and_jacobian!!(
            jprep0, case.x; context=ov.context
        )
    else
        jprep = prepare_fn(
            adtype, _jac_ctx_problem, case.x_proto; check_dims, context=case.context
        )
        @test AbstractPPL.value_and_jacobian!!(jprep, case.x)[2] ≈
            _jac_ctx_expected(case.context, case.x) atol = atol rtol = rtol
        if jacobian_override === :reject
            @test_throws ArgumentError AbstractPPL.value_and_jacobian!!(
                jprep, case.x; context=ov.context
            )
        else
            @test AbstractPPL.value_and_jacobian!!(jprep, case.x; context=ov.context)[2] ≈
                _jac_ctx_expected(ov.context, case.x) atol = atol rtol = rtol
            # Per-call: the frozen context is restored afterwards.
            @test AbstractPPL.value_and_jacobian!!(jprep, case.x)[2] ≈
                _jac_ctx_expected(case.context, case.x) atol = atol rtol = rtol
        end
    end

    # --- override contract violations (validated centrally, all backends) ---
    # Arity/type mismatches and non-`Tuple` overrides throw an
    # `ArgumentError` from `Evaluators._resolve_context` before any backend
    # machinery runs (compiled-tape ReverseDiff throws its own `ArgumentError`
    # even earlier, so the assertions hold on every backend).
    @test_throws ArgumentError AbstractPPL.value_and_gradient!!(
        prep, case.x; context=(ov.context..., ov.context[end])
    )
    @test_throws ArgumentError AbstractPPL.value_and_gradient!!(
        prep, case.x; context=first(ov.context)
    )
    for (i, c) in enumerate(case.context)
        c isa AbstractArray || continue
        bad_eltype = ntuple(
            j -> j == i ? Float32.(case.context[j]) : case.context[j], length(case.context)
        )
        @test_throws ArgumentError AbstractPPL.value_and_gradient!!(
            prep, case.x; context=bad_eltype
        )
    end

    # --- backend-independent invariants ---
    # A stray `context=` on a wrong-arity prep surfaces the domain error, not a
    # MethodError (reject methods accept and ignore `context`). `VectorValuedProblem`
    # is the canonical vector-output (wrong-arity) fixture.
    vprep = prepare_fn(adtype, VectorValuedProblem(), case.x_proto; check_dims)
    @test_throws ArgumentError AbstractPPL.value_and_gradient!!(
        vprep, case.x; context=ov.context
    )

    # Empty input accepts an override on every entry point — no tape/cache built.
    e0 = similar(case.x, 0)
    empty_val = case.f(e0, ov.context...)
    epg = prepare_fn(adtype, case.f, e0; check_dims, context=case.context)
    @test AbstractPPL.value_and_gradient!!(epg, e0; context=ov.context)[1] ≈ empty_val atol =
        atol rtol = rtol
    eph = prepare_fn(adtype, case.f, e0; check_dims, context=case.context, order=2)
    @test AbstractPPL.value_gradient_and_hessian!!(eph, e0; context=ov.context)[1] ≈
        empty_val atol = atol rtol = rtol
    # A non-`Tuple` override is rejected on the empty-input shortcut too.
    @test_throws ArgumentError AbstractPPL.value_and_gradient!!(
        epg, e0; context=first(ov.context)
    )

    # Integer-eltype rejection is preserved under an override: the override path
    # calls `f` directly, so it must apply the same static guard the evaluator
    # callables do. `check_dims=false` isolates the guard (the `{true}` entry
    # check already rejects integers before the override is reached).
    epg_f = prepare_fn(adtype, case.f, e0; check_dims=false, context=case.context)
    @test_throws ArgumentError AbstractPPL.value_and_gradient!!(
        epg_f, Int[]; context=ov.context
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
