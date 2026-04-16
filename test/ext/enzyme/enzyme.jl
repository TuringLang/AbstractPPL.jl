using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using Enzyme
using Test

include(joinpath(@__DIR__, "..", "..", "test_utils.jl"))

struct QuadraticProblem end
struct QuadraticPrepared end

function AbstractPPL.prepare(::QuadraticProblem, x::AbstractVector{<:AbstractFloat})
    return QuadraticPrepared()
end

function (::QuadraticPrepared)(x::AbstractVector{<:AbstractFloat})
    return sum(xi -> xi^2, x)
end

@testset "AbstractPPLEnzymeExt" begin
    problem = QuadraticProblem()
    x0 = zeros(3)
    prepared = AbstractPPL.prepare(ADTypes.AutoEnzyme(), problem, x0)

    @test AbstractPPL.capabilities(prepared) >= AbstractPPL.DerivativeOrder{1}()
    @test AbstractPPL.dimension(prepared) == 3

    x = [3.0, 1.0, 2.0]
    @test prepared(x) ≈ 14.0

    val, grad = AbstractPPL.value_and_gradient(prepared, x)
    @test val ≈ 14.0
    @test grad ≈ [6.0, 2.0, 4.0]
    test_autograd(prepared, x)

    @test_throws DimensionMismatch prepared([3.0, 1.0, 2.0, 99.0])
    @test_throws MethodError prepared([3, 1, 2])
    @test_throws DimensionMismatch AbstractPPL.value_and_gradient(
        prepared, [3.0, 1.0, 2.0, 3.0]
    )
end
