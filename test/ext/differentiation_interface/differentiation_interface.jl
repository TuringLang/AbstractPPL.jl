using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using DifferentiationInterface
using FiniteDifferences
using Test

# Use a backend without a native AbstractPPL extension so this test exercises
# AbstractPPLDifferentiationInterfaceExt dispatch directly.
const fdm = FiniteDifferences.central_fdm(5, 1)
struct DummyADType <: ADTypes.AbstractADType end
const adtype = DummyADType()

struct QuadraticProblem end
struct QuadraticPrepared end

function AbstractPPL.prepare(::QuadraticProblem, x::AbstractVector{<:AbstractFloat})
    return QuadraticPrepared()
end

function (::QuadraticPrepared)(x::AbstractVector{<:AbstractFloat})
    return sum(xi -> xi^2, x)
end

function AbstractPPL.ADProblems.prepare_for_test_autograd(prepared, x::AbstractVector)
    prepared isa typeof(AbstractPPL.prepare(adtype, QuadraticProblem(), x)) ||
        return invoke(
            AbstractPPL.ADProblems.prepare_for_test_autograd, Tuple{Any,Any}, prepared, x
        )
    return (QuadraticProblem(), x, fdm)
end

function DifferentiationInterface.prepare_gradient(f, ::DummyADType, x)
    return DifferentiationInterface.prepare_gradient(
        f, ADTypes.AutoFiniteDifferences(; fdm), x
    )
end

function DifferentiationInterface.value_and_gradient(f, prep, ::DummyADType, x)
    return DifferentiationInterface.value_and_gradient(
        f, prep, ADTypes.AutoFiniteDifferences(; fdm), x
    )
end

@testset "AbstractPPLDifferentiationInterfaceExt" begin
    problem = QuadraticProblem()
    x0 = zeros(3)
    prepared = AbstractPPL.prepare(adtype, problem, x0)

    @test AbstractPPL.capabilities(prepared) >= AbstractPPL.DerivativeOrder{1}()
    @test AbstractPPL.dimension(prepared) == 3

    x = [3.0, 1.0, 2.0]
    @test prepared(x) ≈ 14.0

    val, grad = AbstractPPL.value_and_gradient(prepared, x)
    @test val ≈ 14.0 atol = 1e-6
    @test grad ≈ [6.0, 2.0, 4.0] atol = 1e-6
    test_autograd(prepared, x; atol=1e-4, rtol=1e-4)

    @test_throws DimensionMismatch prepared([3.0, 1.0, 2.0, 99.0])
    @test_throws MethodError prepared([3, 1, 2])
end
