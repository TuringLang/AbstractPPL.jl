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

function AbstractPPL.prepare(::QuadraticProblem, prototype::NamedTuple)
    return QuadraticPrepared()
end

function (::QuadraticPrepared)(values::NamedTuple{(:x, :y)})
    return values.x^2 + sum(vi -> vi^2, values.y)
end

# FiniteDifferences has no native AbstractPPL extension, so AbstractPPLDifferentiationInterfaceExt
# is the only applicable dispatch path for this backend.
const fdm = FiniteDifferences.central_fdm(5, 1)
const adtype = ADTypes.AutoFiniteDifferences(; fdm)

@testset "AbstractPPLDifferentiationInterfaceExt" begin
    problem = QuadraticProblem()
    prototype = (x=0.0, y=[0.0, 0.0])
    prepared = AbstractPPL.prepare(adtype, problem, prototype)

    @test AbstractPPL.capabilities(prepared) >= AbstractPPL.DerivativeOrder{1}()
    @test AbstractPPL.dimension(prepared) == 3

    values = (x=3.0, y=[1.0, 2.0])
    @test prepared(values) ≈ 14.0
    @test prepared([3.0, 1.0, 2.0]) ≈ 14.0

    val, grad = AbstractPPL.value_and_gradient(prepared, values)
    @test val ≈ 14.0 atol=1e-6
    @test grad.x ≈ 6.0 atol=1e-6
    @test grad.y ≈ [2.0, 4.0] atol=1e-6
    test_autograd(prepared, values; atol=1e-4, rtol=1e-4)

    # Overlong vector is rejected
    @test_throws DimensionMismatch prepared([3.0, 1.0, 2.0, 99.0])
end
