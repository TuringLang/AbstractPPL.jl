module AbstractPPLTestExt

using AbstractPPL: AbstractPPL, generate_testcases, run_testcases
using Test: @inferred, @test, @test_broken, @test_throws, @testset

struct QuadraticProblem end
(::QuadraticProblem)(x::AbstractVector{<:Real}) = sum(xi -> xi^2, x)

struct VectorValuedProblem end
(::VectorValuedProblem)(x::AbstractVector{<:Real}) = [x[1] * x[2], x[2] + x[3]]

# Allocation-free vector-output problem for the `:allocations` group:
# `VectorValuedProblem` allocates its result vector, masking AD-path allocations.
struct IdentityProblem end
(::IdentityProblem)(x::AbstractVector{<:Real}) = x

struct ValueCase
    name::String
    f::Any
    x_proto::Any
    x::Any
    value::Any
    gradient::Any
    jacobian::Any
end

struct HessianCase
    name::String
    f::Any
    x_proto::Any
    x::Any
    value::Any
    gradient::Any
    hessian::Any
end

struct ErrorCase
    name::String
    f::Any
    x_proto::Any
    x::Any
    op::Any
    exception::Any
end

function AbstractPPL.generate_testcases(::Val{:vector})
    return (
        ValueCase(
            "quadratic (scalar output)",
            QuadraticProblem(),
            zeros(3),
            [3.0, 1.0, 2.0],
            14.0,
            [6.0, 2.0, 4.0],
            nothing,
        ),
        ValueCase(
            "vector-valued (vector output)",
            VectorValuedProblem(),
            zeros(3),
            [2.0, 3.0, 4.0],
            [6.0, 7.0],
            nothing,
            [3.0 2.0 0.0; 0.0 1.0 1.0],
        ),
        ValueCase(
            "empty input, scalar output",
            x -> 7.5,
            Float64[],
            Float64[],
            7.5,
            Float64[],
            nothing,
        ),
        ValueCase(
            "empty input, vector output",
            x -> [2.0, 3.0],
            Float64[],
            Float64[],
            [2.0, 3.0],
            nothing,
            zeros(2, 0),
        ),
    )
end

function AbstractPPL.generate_testcases(::Val{:hessian})
    return (
        HessianCase(
            "quadratic (scalar output)",
            QuadraticProblem(),
            zeros(3),
            [3.0, 1.0, 2.0],
            14.0,
            [6.0, 2.0, 4.0],
            [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0],
        ),
        HessianCase(
            "empty input, scalar output",
            x -> 7.5,
            Float64[],
            Float64[],
            7.5,
            Float64[],
            zeros(0, 0),
        ),
    )
end

function AbstractPPL.generate_testcases(::Val{:hessian_edge})
    return (
        # `value_gradient_and_hessian!!` rejects order=1 preps regardless of
        # the underlying problem arity — both paths share the same dispatch
        # so one case suffices.
        ErrorCase(
            "value_gradient_and_hessian!! on order=1 prep",
            QuadraticProblem(),
            zeros(3),
            [3.0, 1.0, 2.0],
            (prepared, x) -> AbstractPPL.value_gradient_and_hessian!!(prepared, x),
            r"order=2",
        ),
    )
end

function AbstractPPL.generate_testcases(::Val{:edge})
    return (
        ErrorCase(
            "wrong vector length",
            QuadraticProblem(),
            zeros(3),
            [3.0, 1.0, 2.0, 99.0],
            (prepared, x) -> prepared(x),
            DimensionMismatch,
        ),
        ErrorCase(
            "non-floating-point vector",
            QuadraticProblem(),
            zeros(3),
            [3, 1, 2],
            (prepared, x) -> prepared(x),
            r"floating-point",
        ),
        ErrorCase(
            "gradient of vector-valued output",
            VectorValuedProblem(),
            zeros(3),
            [2.0, 3.0, 4.0],
            (prepared, x) -> AbstractPPL.value_and_gradient!!(prepared, x),
            r"scalar-valued",
        ),
        ErrorCase(
            "jacobian of scalar output",
            QuadraticProblem(),
            zeros(3),
            [3.0, 1.0, 2.0],
            (prepared, x) -> AbstractPPL.value_and_jacobian!!(prepared, x),
            r"vector-valued",
        ),
        ErrorCase(
            "gradient of vector-valued output, empty input",
            x -> [2.0, 3.0],
            Float64[],
            Float64[],
            (prepared, x) -> AbstractPPL.value_and_gradient!!(prepared, x),
            r"scalar-valued",
        ),
        ErrorCase(
            "jacobian of scalar output, empty input",
            x -> 7.5,
            Float64[],
            Float64[],
            (prepared, x) -> AbstractPPL.value_and_jacobian!!(prepared, x),
            r"vector-valued",
        ),
        ErrorCase(
            "value_and_gradient!! wrong vector length",
            QuadraticProblem(),
            zeros(3),
            [3.0, 1.0, 2.0, 99.0],
            (prepared, x) -> AbstractPPL.value_and_gradient!!(prepared, x),
            DimensionMismatch,
        ),
        ErrorCase(
            "value_and_jacobian!! wrong vector length",
            VectorValuedProblem(),
            zeros(3),
            [2.0, 3.0, 4.0, 5.0],
            (prepared, x) -> AbstractPPL.value_and_jacobian!!(prepared, x),
            DimensionMismatch,
        ),
        ErrorCase(
            "value_and_gradient!! non-floating-point vector",
            QuadraticProblem(),
            zeros(3),
            [3, 1, 2],
            (prepared, x) -> AbstractPPL.value_and_gradient!!(prepared, x),
            r"floating-point",
        ),
        ErrorCase(
            "value_and_jacobian!! non-floating-point vector",
            VectorValuedProblem(),
            zeros(3),
            [2, 3, 4],
            (prepared, x) -> AbstractPPL.value_and_jacobian!!(prepared, x),
            r"floating-point",
        ),
    )
end

function AbstractPPL.generate_testcases(::Val{:namedtuple})
    return (
        ValueCase(
            "scalar output over (x::Real, y::Vector)",
            vs -> vs.x^2 + sum(abs2, vs.y),
            (x=0.0, y=zeros(2)),
            (x=3.0, y=[1.0, 2.0]),
            14.0,
            (x=6.0, y=[2.0, 4.0]),
            nothing,
        ),
    )
end

function AbstractPPL.run_testcases(
    ::Val{:vector}, prepare_fn=AbstractPPL.prepare; adtype, atol=0, rtol=1e-10
)
    for case in generate_testcases(Val(:vector))
        @testset "$(case.name)" begin
            prepared = prepare_fn(adtype, case.f, case.x_proto)
            @test AbstractPPL.order(prepared) == 1
            @test prepared(case.x) ≈ case.value atol = atol rtol = rtol
            if case.gradient !== nothing
                val, grad = AbstractPPL.value_and_gradient!!(prepared, case.x)
                @test val ≈ case.value atol = atol rtol = rtol
                @test grad ≈ case.gradient atol = atol rtol = rtol
            end
            if case.jacobian !== nothing
                val, jac = AbstractPPL.value_and_jacobian!!(prepared, case.x)
                @test val ≈ case.value atol = atol rtol = rtol
                @test jac ≈ case.jacobian atol = atol rtol = rtol
            end
        end
    end
    return nothing
end

function AbstractPPL.run_testcases(
    ::Val{:hessian}, prepare_fn=AbstractPPL.prepare; adtype, atol=0, rtol=1e-10
)
    for case in generate_testcases(Val(:hessian))
        @testset "$(case.name)" begin
            prepared = prepare_fn(adtype, case.f, case.x_proto; order=2)
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
        end
    end
    for case in generate_testcases(Val(:hessian_edge))
        @testset "$(case.name)" begin
            prepared = prepare_fn(adtype, case.f, case.x_proto)
            @test_throws case.exception case.op(prepared, case.x)
        end
    end
    return nothing
end

function AbstractPPL.run_testcases(::Val{:edge}, prepare_fn=AbstractPPL.prepare; adtype)
    for case in generate_testcases(Val(:edge))
        @testset "$(case.name)" begin
            prepared = prepare_fn(adtype, case.f, case.x_proto)
            @test_throws case.exception case.op(prepared, case.x)
        end
    end
    return nothing
end

function AbstractPPL.run_testcases(
    ::Val{:namedtuple}, prepare_fn=AbstractPPL.prepare; adtype, atol=0, rtol=1e-10
)
    for case in generate_testcases(Val(:namedtuple))
        @testset "$(case.name)" begin
            prepared = prepare_fn(adtype, case.f, case.x_proto)
            @test prepared(case.x) ≈ case.value atol = atol rtol = rtol
            if case.gradient !== nothing
                val, grad = AbstractPPL.value_and_gradient!!(prepared, case.x)
                @test val ≈ case.value atol = atol rtol = rtol
                for k in keys(case.gradient)
                    @test getproperty(grad, k) ≈ getproperty(case.gradient, k) atol = atol rtol =
                        rtol
                end
            end
        end
    end
    return nothing
end

# Drive `value_and_{gradient,jacobian}!!` twice with different inputs against
# the same `prepared` evaluator to exercise cache reuse — catches backends
# whose cache state is corrupted by a prior call.
function AbstractPPL.run_testcases(
    ::Val{:cache_reuse}, prepare_fn=AbstractPPL.prepare; adtype, atol=0, rtol=1e-10
)
    @testset "scalar output, repeated calls" begin
        prepared = prepare_fn(adtype, QuadraticProblem(), zeros(3))
        for (x, value, gradient) in (
            ([1.0, 2.0, 3.0], 14.0, [2.0, 4.0, 6.0]),
            ([4.0, 5.0, 6.0], 77.0, [8.0, 10.0, 12.0]),
            ([0.5, -1.0, 2.0], 5.25, [1.0, -2.0, 4.0]),
        )
            val, grad = AbstractPPL.value_and_gradient!!(prepared, x)
            @test val ≈ value atol = atol rtol = rtol
            @test grad ≈ gradient atol = atol rtol = rtol
        end
    end
    @testset "vector output, repeated calls" begin
        prepared = prepare_fn(adtype, VectorValuedProblem(), zeros(3))
        for (x, value, jacobian) in (
            ([2.0, 3.0, 4.0], [6.0, 7.0], [3.0 2.0 0.0; 0.0 1.0 1.0]),
            ([5.0, 1.0, 7.0], [5.0, 8.0], [1.0 5.0 0.0; 0.0 1.0 1.0]),
            ([0.0, 4.0, -2.0], [0.0, 2.0], [4.0 0.0 0.0; 0.0 1.0 1.0]),
        )
            val, jac = AbstractPPL.value_and_jacobian!!(prepared, x)
            @test val ≈ value atol = atol rtol = rtol
            @test jac ≈ jacobian atol = atol rtol = rtol
        end
    end
    return nothing
end

# Helpers for the `:type_stability` group: `@inferred` is a syntactic macro, so wrap
# each AD entry in a tiny named function that returns `true` on success — that
# value lets `@test` / `@test_broken` evaluate the call uniformly.
function _inferred_gradient(prepared, x)
    return (@inferred AbstractPPL.value_and_gradient!!(prepared, x); true)
end
function _inferred_jacobian(prepared, x)
    return (@inferred AbstractPPL.value_and_jacobian!!(prepared, x); true)
end
function _inferred_hessian(prepared, x)
    return (@inferred AbstractPPL.value_gradient_and_hessian!!(prepared, x); true)
end

# Backends with known regressions (e.g. Mooncake's allocating
# `value_and_jacobian!!`, or its forward-mode Jacobian inference) pass
# `*_broken=true` to mark the assertion as broken instead of failing.
function AbstractPPL.run_testcases(
    ::Val{:allocations},
    prepare_fn=AbstractPPL.prepare;
    adtype,
    gradient_broken::Bool=false,
    jacobian_broken::Bool=false,
)
    x = [1.0, 2.0, 3.0]
    @testset "scalar gradient" begin
        prepared = prepare_fn(adtype, QuadraticProblem(), zeros(3); check_dims=false)
        AbstractPPL.value_and_gradient!!(prepared, x)  # warm up
        allocs = @allocated AbstractPPL.value_and_gradient!!(prepared, x)
        if gradient_broken
            @test_broken allocs == 0
        else
            @test allocs == 0
        end
    end
    @testset "vector jacobian" begin
        prepared = prepare_fn(adtype, IdentityProblem(), zeros(3); check_dims=false)
        AbstractPPL.value_and_jacobian!!(prepared, x)
        allocs = @allocated AbstractPPL.value_and_jacobian!!(prepared, x)
        if jacobian_broken
            @test_broken allocs == 0
        else
            @test allocs == 0
        end
    end
    return nothing
end

function AbstractPPL.run_testcases(
    ::Val{:type_stability},
    prepare_fn=AbstractPPL.prepare;
    adtype,
    gradient_broken::Bool=false,
    jacobian_broken::Bool=false,
    hessian_broken::Bool=false,
)
    x = [1.0, 2.0, 3.0]
    @testset "scalar gradient" begin
        prepared = prepare_fn(adtype, QuadraticProblem(), zeros(3); check_dims=false)
        if gradient_broken
            @test_broken _inferred_gradient(prepared, x)
        else
            @test _inferred_gradient(prepared, x)
        end
    end
    @testset "vector jacobian" begin
        prepared = prepare_fn(adtype, IdentityProblem(), zeros(3); check_dims=false)
        if jacobian_broken
            @test_broken _inferred_jacobian(prepared, x)
        else
            @test _inferred_jacobian(prepared, x)
        end
    end
    @testset "hessian" begin
        prepared = prepare_fn(
            adtype, QuadraticProblem(), zeros(3); check_dims=false, order=2
        )
        if hessian_broken
            @test_broken _inferred_hessian(prepared, x)
        else
            @test _inferred_hessian(prepared, x)
        end
    end
    return nothing
end

function AbstractPPL.run_testcases(
    ::Val{:context}, prepare_fn=AbstractPPL.prepare; adtype, atol=0, rtol=1e-10
)
    # `prepare(adtype, f, x; context=(c,))` builds an evaluator that computes
    # `f(x, context...)` with AD differentiating only `x`.
    f(y::AbstractVector{<:Real}, offset) = -0.5 * (y[1] - offset)^2
    x = [0.3]
    c = 0.1
    @testset "scalar gradient with context" begin
        prepared = prepare_fn(adtype, f, x; check_dims=false, context=(c,))
        @test prepared(x) ≈ f(x, c) atol = atol rtol = rtol
        val, grad = AbstractPPL.value_and_gradient!!(prepared, x)
        @test val ≈ f(x, c) atol = atol rtol = rtol
        @test grad ≈ [-(x[1] - c)] atol = atol rtol = rtol
    end
    return nothing
end

end # module
