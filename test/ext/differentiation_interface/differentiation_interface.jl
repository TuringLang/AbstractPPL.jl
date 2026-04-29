using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using DifferentiationInterface
using Test

include(joinpath(@__DIR__, "..", "ad_tests.jl"))

# Use a backend without a native AbstractPPL extension so this test exercises
# AbstractPPLDifferentiationInterfaceExt dispatch directly.
struct DummyADType <: ADTypes.AbstractADType end
const adtype = DummyADType()

function DifferentiationInterface.prepare_gradient(
    f, ::DummyADType, x, ctx::DifferentiationInterface.Constant
)
    return Val(:gradient)
end

function DifferentiationInterface.value_and_gradient(
    f, prep, ::DummyADType, x, ctx::DifferentiationInterface.Constant
)
    return (f(x, ctx.data), 2 .* x)
end

function DifferentiationInterface.prepare_jacobian(
    f, ::DummyADType, x, ctx::DifferentiationInterface.Constant
)
    return Val(:jacobian)
end

function DifferentiationInterface.value_and_jacobian(
    f, prep, ::DummyADType, x, ctx::DifferentiationInterface.Constant
)
    jac = [
        x[2] x[1] zero(eltype(x))
        zero(eltype(x)) one(eltype(x)) one(eltype(x))
    ]
    return (f(x, ctx.data), jac)
end

struct ZeroDimProblem end
(::ZeroDimProblem)(::AbstractVector) = 7.5

struct ZeroDimVecProblem end
(::ZeroDimVecProblem)(::AbstractVector) = [2.0, 3.0]

@testset "AbstractPPLDifferentiationInterfaceExt" begin
    x0 = zeros(3)
    x = [3.0, 1.0, 2.0]

    run_shared_gradient_tests(adtype, x0, x; atol=1e-6, rtol=1e-6)
    run_shared_jacobian_tests(adtype, x0, [2.0, 3.0, 4.0]; atol=1e-6, rtol=1e-6)

    @testset "empty input short-circuits DI" begin
        x_empty = Float64[]

        prepared = AbstractPPL.prepare(adtype, ZeroDimProblem(), x_empty)
        @test prepared isa AbstractPPL.ADProblems.AbstractPrepared
        val, grad = AbstractPPL.value_and_gradient(prepared, x_empty)
        @test val == 7.5
        @test grad == Float64[]

        prepared_jac = AbstractPPL.prepare(adtype, ZeroDimVecProblem(), x_empty)
        valj, jac = AbstractPPL.value_and_jacobian(prepared_jac, x_empty)
        @test valj == [2.0, 3.0]
        @test size(jac) == (2, 0)
    end
end
