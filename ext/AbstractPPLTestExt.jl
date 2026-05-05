module AbstractPPLTestExt

using AbstractPPL: AbstractPPL, generate_testcases, run_testcases
using Test: @test, @test_throws, @testset

struct QuadraticProblem end
(::QuadraticProblem)(x::AbstractVector{<:Real}) = sum(xi -> xi^2, x)

struct VectorValuedProblem end
(::VectorValuedProblem)(x::AbstractVector{<:Real}) = [x[1] * x[2], x[2] + x[3]]

struct ValueCase
    name::String
    f::Any
    x_proto::Any
    x::Any
    value::Any
    gradient::Any
    jacobian::Any
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
            Exception,
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
    )
end

function AbstractPPL.run_testcases(
    ::Val{:vector}, prepare_fn=AbstractPPL.prepare; adtype, atol=0, rtol=1e-10
)
    for case in generate_testcases(Val(:vector))
        @testset "$(case.name)" begin
            prepared = prepare_fn(adtype, case.f, case.x_proto)
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

function AbstractPPL.run_testcases(::Val{:edge}, prepare_fn=AbstractPPL.prepare; adtype)
    for case in generate_testcases(Val(:edge))
        @testset "$(case.name)" begin
            prepared = prepare_fn(adtype, case.f, case.x_proto)
            @test_throws case.exception case.op(prepared, case.x)
        end
    end
    return nothing
end

end # module
