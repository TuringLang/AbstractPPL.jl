using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using FiniteDifferences
using Test

include(joinpath(@__DIR__, "..", "..", "test_utils.jl"))

struct QuadraticProblem end
struct QuadraticPrepared end

function AbstractPPL.prepare(::QuadraticProblem, values::NamedTuple)
    return QuadraticPrepared()
end

function (::QuadraticPrepared)(values::NamedTuple{(:x, :y)})
    return values.x^2 + sum(vi -> vi^2, values.y)
end

@testset "AbstractPPLFiniteDifferencesExt" begin
    problem = QuadraticProblem()
    values = (x=0.0, y=[0.0, 0.0])
    fdm = FiniteDifferences.central_fdm(5, 1)
    prepared = AbstractPPL.prepare(ADTypes.AutoFiniteDifferences(; fdm), problem, values)

    @test AbstractPPL.capabilities(prepared) >= AbstractPPL.DerivativeOrder{1}()
    @test AbstractPPL.dimension(prepared) == 3

    values = (x=3.0, y=[1.0, 2.0])
    @test prepared(values) ≈ 14.0
    @test prepared([3.0, 1.0, 2.0]) ≈ 14.0

    val, grad = AbstractPPL.value_and_gradient(prepared, values)
    @test val ≈ 14.0
    test_autograd(prepared, values)

    @test_throws DimensionMismatch prepared([3.0, 1.0, 2.0, 99.0])
    @test_throws MethodError prepared([3, 1, 2])
    @test_throws MethodError prepared((x=3.0, z=[1.0, 2.0]))
    @test_throws DimensionMismatch AbstractPPL.value_and_gradient(
        prepared, (x=3.0, y=[1.0, 2.0, 3.0])
    )
end
