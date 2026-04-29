# Shared problem definitions and test helpers for AD backend integration tests.
# Include this file after `using AbstractPPL, Test` and any backend-specific setup.

struct QuadraticProblem end
struct QuadraticVecPrepared end

function AbstractPPL.prepare(::QuadraticProblem, x::AbstractVector{<:Real})
    return QuadraticVecPrepared()
end

function (::QuadraticVecPrepared)(x::AbstractVector{<:Real})
    return sum(xi -> xi^2, x)
end

struct VectorValuedProblem end
struct VectorValuedPrepared end

function AbstractPPL.prepare(::VectorValuedProblem, x::AbstractVector{<:Real})
    return VectorValuedPrepared()
end

# y = [x[1]*x[2], x[2]+x[3]] -> J = [x[2] x[1] 0; 0 1 1]
function (::VectorValuedPrepared)(x::AbstractVector{<:Real})
    return [x[1] * x[2], x[2] + x[3]]
end

"""
    run_shared_gradient_tests(adtype, x0, x; atol=0, rtol=1e-10)

Test the vector-input gradient path for `adtype` on `QuadraticProblem`.
`x0` is the prototype (zeros), `x = [3.0, 1.0, 2.0]` is the test point.
"""
function run_shared_gradient_tests(adtype, x0, x; atol=0, rtol=1e-10)
    @testset "gradient path" begin
        problem = QuadraticProblem()
        prepared = AbstractPPL.prepare(adtype, problem, x0)

        @test prepared(x) ≈ 14.0

        val, grad = AbstractPPL.value_and_gradient(prepared, x)
        @test val ≈ 14.0 atol = atol rtol = rtol
        @test grad ≈ [6.0, 2.0, 4.0] atol = atol rtol = rtol

        @test_throws DimensionMismatch prepared([3.0, 1.0, 2.0, 99.0])
        @test_throws r"floating-point" prepared([3, 1, 2])
    end
end

"""
    run_shared_jacobian_tests(adtype, x0, xj; atol=0, rtol=1e-10)

Test the jacobian path for `adtype` on `VectorValuedProblem`.
`x0` is the prototype (zeros(3)), `xj` is the test point.
"""
function run_shared_jacobian_tests(adtype, x0, xj; atol=0, rtol=1e-10)
    @testset "jacobian path" begin
        problem = VectorValuedProblem()
        prepared = AbstractPPL.prepare(adtype, problem, x0)

        @test prepared(xj) ≈ [6.0, 7.0]

        val, jac = AbstractPPL.value_and_jacobian(prepared, xj)
        @test val ≈ [6.0, 7.0] atol = atol rtol = rtol
        @test jac ≈ [3.0 2.0 0.0; 0.0 1.0 1.0] atol = atol rtol = rtol

        @test_throws r"scalar-valued" AbstractPPL.value_and_gradient(prepared, xj)
    end
end
