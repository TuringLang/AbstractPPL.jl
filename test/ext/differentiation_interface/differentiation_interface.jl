using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using DifferentiationInterface
using FiniteDifferences
using Test

include(joinpath(@__DIR__, "..", "..", "test_utils.jl"))

struct QuadraticProblem end
struct QuadraticPrepared end
struct DummyADType <: ADTypes.AbstractADType end

function AbstractPPL.prepare(::QuadraticProblem, values::NamedTuple)
    return QuadraticPrepared()
end

function (::QuadraticPrepared)(values::NamedTuple{(:x, :y)})
    return values.x^2 + sum(vi -> vi^2, values.y)
end

# Use a backend without a native AbstractPPL extension so this test exercises
# AbstractPPLDifferentiationInterfaceExt dispatch directly.
const fdm = FiniteDifferences.central_fdm(5, 1)
const adtype = DummyADType()

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
    values = (x=0.0, y=[0.0, 0.0])
    prepared = AbstractPPL.prepare(adtype, problem, values)

    @test AbstractPPL.capabilities(prepared) >= AbstractPPL.DerivativeOrder{1}()
    @test AbstractPPL.dimension(prepared) == 3

    values = (x=3.0, y=[1.0, 2.0])
    @test prepared(values) ≈ 14.0
    @test prepared([3.0, 1.0, 2.0]) ≈ 14.0

    val, grad = AbstractPPL.value_and_gradient(prepared, values)
    @test val ≈ 14.0 atol = 1e-6
    @test grad.x ≈ 6.0 atol = 1e-6
    @test grad.y ≈ [2.0, 4.0] atol = 1e-6
    test_autograd(prepared, values; atol=1e-4, rtol=1e-4)

    # Overlong vector is rejected
    @test_throws DimensionMismatch prepared([3.0, 1.0, 2.0, 99.0])
    @test_throws MethodError prepared([3, 1, 2])
    @test_throws MethodError prepared((x=3.0, z=[1.0, 2.0]))
    @test_throws DimensionMismatch AbstractPPL.value_and_gradient(
        prepared, (x=3.0, y=[1.0, 2.0, 3.0])
    )
end
