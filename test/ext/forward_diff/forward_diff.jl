using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using FiniteDifferences
using ForwardDiff
using LogDensityProblems: LogDensityProblems
using Test

include(joinpath(@__DIR__, "..", "ad_tests.jl"))

struct RequestedTag end

@testset "AbstractPPLForwardDiffExt" begin
    adtype = ADTypes.AutoForwardDiff()
    x0 = zeros(3)
    x = [3.0, 1.0, 2.0]
    values0 = (x=0.0, y=[0.0, 0.0])
    values = (x=3.0, y=[1.0, 2.0])

    run_shared_gradient_tests(adtype, x0, x)
    run_shared_jacobian_tests(adtype, x0, [2.0, 3.0, 4.0])
    run_shared_namedtuple_tests(adtype, values0, values)
    run_shared_invalid_mode_tests(adtype, x0)
    run_shared_ldp_tests(adtype, x0, x)

    @testset "check_dims=false skips dim/shape checks" begin
        problem = QuadraticProblem()
        prepared_unchecked = AbstractPPL.prepare(adtype, problem, x0; check_dims=false)
        val, grad = AbstractPPL.value_and_gradient(prepared_unchecked, x)
        @test val ≈ 14.0
        @test grad ≈ [6.0, 2.0, 4.0]

        prepared_nt_unchecked = AbstractPPL.prepare(
            adtype, problem, values0; check_dims=false
        )
        @test prepared_nt_unchecked(values) ≈ 14.0
    end

    @testset "honors caller-provided custom tag" begin
        ad = ADTypes.AutoForwardDiff(;
            chunksize=1, tag=ForwardDiff.Tag(RequestedTag(), Float64)
        )
        problem = QuadraticProblem()
        prepared_tagged = AbstractPPL.prepare(ad, problem, x0)
        val, grad = AbstractPPL.value_and_gradient(prepared_tagged, x)
        @test val ≈ 14.0
        @test grad ≈ [6.0, 2.0, 4.0]
    end
end
