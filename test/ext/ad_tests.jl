# Shared problem definitions and test helpers for AD backend integration tests.
# Include this file after `using AbstractPPL, Test` and any backend-specific setup.

struct QuadraticProblem end
struct QuadraticNTPrepared end
struct QuadraticVecPrepared end

function AbstractPPL.prepare(::QuadraticProblem, values::NamedTuple)
    return QuadraticNTPrepared()
end

function (::QuadraticNTPrepared)(values::NamedTuple{(:x, :y)})
    return values.x^2 + sum(vi -> vi^2, values.y)
end

function AbstractPPL.prepare(::QuadraticProblem, x::AbstractVector{<:AbstractFloat})
    return QuadraticVecPrepared()
end

function (::QuadraticVecPrepared)(x::AbstractVector{<:Real})
    return sum(xi -> xi^2, x)
end

struct VectorValuedProblem end
struct VectorValuedPrepared end

function AbstractPPL.prepare(::VectorValuedProblem, x::AbstractVector{<:AbstractFloat})
    return VectorValuedPrepared()
end

# y = [x[1]*x[2], x[2]+x[3]] -> J = [x[2] x[1] 0; 0 1 1]
function (::VectorValuedPrepared)(x::AbstractVector{<:Real})
    return [x[1] * x[2], x[2] + x[3]]
end

"""
    run_shared_gradient_tests(adtype, x0, x; atol=0, rtol=1e-10, test_autograd_kwargs=NamedTuple())

Test the vector-input gradient path for `adtype` on `QuadraticProblem`.
`x0` is the prototype (zeros), `x = [3.0, 1.0, 2.0]` is the test point.
"""
function run_shared_gradient_tests(
    adtype, x0, x; atol=0, rtol=1e-10, test_autograd_kwargs=NamedTuple()
)
    @testset "gradient path" begin
        problem = QuadraticProblem()
        prepared = AbstractPPL.prepare(adtype, problem, x0)

        @test AbstractPPL.dimension(prepared) == length(x0)

        @test prepared(x) ≈ 14.0

        val, grad = AbstractPPL.value_and_gradient(prepared, x)
        @test val ≈ 14.0 atol = atol rtol = rtol
        @test grad ≈ [6.0, 2.0, 4.0] atol = atol rtol = rtol
        test_autograd(prepared, x; test_autograd_kwargs...)

        @test_throws DimensionMismatch prepared([3.0, 1.0, 2.0, 99.0])
        @test_throws MethodError prepared([3, 1, 2])
    end
end

"""
    run_shared_jacobian_tests(adtype, x0, xj; atol=0, rtol=1e-10, test_autograd_kwargs=NamedTuple())

Test the jacobian path for `adtype` on `VectorValuedProblem`.
`x0` is the prototype (zeros(3)), `xj` is the test point.
"""
function run_shared_jacobian_tests(
    adtype, x0, xj; atol=0, rtol=1e-10, test_autograd_kwargs=NamedTuple()
)
    @testset "jacobian path" begin
        problem = VectorValuedProblem()
        prepared = AbstractPPL.prepare(adtype, problem, x0; mode=:jacobian)

        @test AbstractPPL.dimension(prepared) == length(x0)

        @test prepared(xj) ≈ [6.0, 7.0]

        val, jac = AbstractPPL.value_and_jacobian(prepared, xj)
        @test val ≈ [6.0, 7.0] atol = atol rtol = rtol
        @test jac ≈ [3.0 2.0 0.0; 0.0 1.0 1.0] atol = atol rtol = rtol

        @test_throws ArgumentError AbstractPPL.value_and_gradient(prepared, xj)
        @test_throws r"only supports gradient-mode" test_autograd(
            prepared, xj; test_autograd_kwargs...
        )
    end
end

"""
    run_shared_namedtuple_tests(adtype, values0, values; atol=0, rtol=1e-10)

Test the NamedTuple-input gradient path for `adtype` on `QuadraticProblem`.
`values0` is the prototype, `values = (x=3.0, y=[1.0, 2.0])` is the test point.
"""
function run_shared_namedtuple_tests(
    adtype, values0, values; atol=0, rtol=1e-10, test_autograd_kwargs=NamedTuple()
)
    @testset "NamedTuple path" begin
        problem = QuadraticProblem()
        prepared = AbstractPPL.prepare(adtype, problem, values0)

        @test_throws r"only available for evaluators prepared with a vector" AbstractPPL.dimension(
            prepared
        )

        @test prepared(values) ≈ 14.0

        val, grad = AbstractPPL.value_and_gradient(prepared, values)
        @test val ≈ 14.0
        @test grad.x ≈ 6.0 atol = atol rtol = rtol
        @test grad.y ≈ [2.0, 4.0] atol = atol rtol = rtol
        test_autograd(prepared, values; test_autograd_kwargs...)

        @test_throws r"same NamedTuple structure" prepared((x=3.0, z=[1.0, 2.0]))
        @test_throws r"same NamedTuple structure" AbstractPPL.value_and_gradient(
            prepared, (x=3.0, y=reshape([1.0, 2.0], 1, 2))
        )

        @test_throws r"only supported on the vector-input path" AbstractPPL.prepare(
            adtype, problem, values0; mode=:jacobian
        )
    end
end

"""
    run_shared_invalid_mode_tests(adtype, x0)

Test that `mode=:hessian` is rejected at prepare-time.
"""
function run_shared_invalid_mode_tests(adtype, x0)
    @testset "invalid mode rejected" begin
        @test_throws r"`mode` must be" AbstractPPL.prepare(
            adtype, QuadraticProblem(), x0; mode=:hessian
        )
    end
end
