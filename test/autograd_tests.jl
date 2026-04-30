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

        val, grad = AbstractPPL.value_and_gradient!!(prepared, x)
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

        val, jac = AbstractPPL.value_and_jacobian!!(prepared, xj)
        @test val ≈ [6.0, 7.0] atol = atol rtol = rtol
        @test jac ≈ [3.0 2.0 0.0; 0.0 1.0 1.0] atol = atol rtol = rtol

        @test_throws r"scalar-valued" AbstractPPL.value_and_gradient!!(prepared, xj)
    end
end

"""
    run_shared_empty_input_tests(adtype)

Test the empty-input short-circuit for `adtype` on both scalar- and
vector-valued evaluators.
"""
function run_shared_empty_input_tests(adtype)
    @testset "empty input" begin
        x_empty = Float64[]
        prepared = AbstractPPL.prepare(adtype, x -> 7.5, x_empty)
        val, grad = AbstractPPL.value_and_gradient!!(prepared, x_empty)
        @test val == 7.5
        @test grad == Float64[]

        prepared_jac = AbstractPPL.prepare(adtype, x -> [2.0, 3.0], x_empty)
        valj, jac = AbstractPPL.value_and_jacobian!!(prepared_jac, x_empty)
        @test valj == [2.0, 3.0]
        @test size(jac) == (2, 0)
    end
end

"""
    run_shared_namedtuple_tests(adtype)

Test the NamedTuple-input gradient path for AD backends that support it
(currently the native ForwardDiff and Mooncake extensions).
"""
function run_shared_namedtuple_tests(adtype)
    @testset "NamedTuple input" begin
        prepared = AbstractPPL.prepare(
            adtype, vs -> vs.x^2 + sum(abs2, vs.y), (x=0.0, y=zeros(2))
        )
        @test prepared.evaluator isa AbstractPPL.ADProblems.NamedTupleEvaluator

        val, grad = AbstractPPL.value_and_gradient!!(prepared, (x=3.0, y=[1.0, 2.0]))
        @test val ≈ 14.0
        @test grad.x ≈ 6.0
        @test grad.y ≈ [2.0, 4.0]

        @test_throws r"same NamedTuple structure" AbstractPPL.value_and_gradient!!(
            prepared, (x=3.0, z=[1.0, 2.0])
        )
    end
end

"""
    run_autograd_tests(adtype; namedtuple=false, extra_jacobian_modes=(), kwargs...)

Run the gradient, jacobian, and empty-input shared tests on `adtype`. Pass
`namedtuple=true` for backends with a native NamedTuple-input path; pass
`extra_jacobian_modes` to additionally exercise the jacobian path on those
adtypes. `kwargs` (`atol`, `rtol`, …) are forwarded to the shared helpers.
"""
function run_autograd_tests(
    adtype; namedtuple::Bool=false, extra_jacobian_modes=(), kwargs...
)
    run_shared_gradient_tests(adtype, zeros(3), [3.0, 1.0, 2.0]; kwargs...)
    run_shared_jacobian_tests(adtype, zeros(3), [2.0, 3.0, 4.0]; kwargs...)
    run_shared_empty_input_tests(adtype)
    for adtype_jac in extra_jacobian_modes
        run_shared_jacobian_tests(adtype_jac, zeros(3), [2.0, 3.0, 4.0]; kwargs...)
    end
    namedtuple && run_shared_namedtuple_tests(adtype)
    return nothing
end
