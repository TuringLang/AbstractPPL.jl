using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using Mooncake
using Test

include(joinpath(@__DIR__, "..", "..", "test_utils.jl"))

struct QuadraticProblem end
struct QuadraticPrepared end

function AbstractPPL.prepare(::QuadraticProblem, prototype::NamedTuple)
    return QuadraticPrepared()
end

function (::QuadraticPrepared)(values::NamedTuple{(:x, :y)})
    return values.x^2 + sum(vi -> vi^2, values.y)
end

@testset "AbstractPPLMooncakeExt" begin
    problem = QuadraticProblem()
    prototype = (x=0.0, y=[0.0, 0.0])
    prepared = AbstractPPL.prepare(
        ADTypes.AutoMooncake(; config=Mooncake.Config()), problem, prototype
    )

    @test AbstractPPL.capabilities(prepared) >= AbstractPPL.DerivativeOrder{1}()
    @test AbstractPPL.dimension(prepared) == 3

    values = (x=3.0, y=[1.0, 2.0])
    @test prepared(values) ≈ 14.0
    @test prepared([3.0, 1.0, 2.0]) ≈ 14.0

    val, grad = AbstractPPL.value_and_gradient(prepared, values)
    @test val ≈ 14.0
    @test grad.x ≈ 6.0
    @test grad.y ≈ [2.0, 4.0]
    test_autograd(prepared, values)

    # Overlong vector is rejected
    @test_throws DimensionMismatch prepared([3.0, 1.0, 2.0, 99.0])
end
