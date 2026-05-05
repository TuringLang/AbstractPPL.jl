using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL: AbstractPPL, prepare, run_testcases
using ADTypes: AutoForwardDiff
using DifferentiationInterface: DifferentiationInterface as DI
using ForwardDiff
using LogDensityProblems: LogDensityProblems
using Test

@testset "AbstractPPLDifferentiationInterfaceExt" begin
    run_testcases(Val(:vector); adtype=AutoForwardDiff(), atol=1e-6, rtol=1e-6)
    run_testcases(Val(:edge); adtype=AutoForwardDiff())

    @testset "LogDensityProblems capabilities" begin
        p_scalar = prepare(AutoForwardDiff(), x -> -0.5 * sum(abs2, x), zeros(3))
        @test LogDensityProblems.capabilities(p_scalar) ==
            LogDensityProblems.LogDensityOrder{1}()
        x = [1.0, 2.0, 3.0]
        val, grad = LogDensityProblems.logdensity_and_gradient(p_scalar, x)
        @test val ≈ -0.5 * sum(abs2, x)
        @test grad ≈ -x

        # All DI-prepared evaluators advertise order 1; mismatched arity
        # surfaces as a runtime error from `value_and_gradient!!` rather
        # than a capability downgrade.
        p_vec = prepare(AutoForwardDiff(), x -> [x[1] * x[2], x[2] + x[3]], zeros(3))
        @test LogDensityProblems.capabilities(p_vec) ==
            LogDensityProblems.LogDensityOrder{1}()

        p_empty = prepare(AutoForwardDiff(), x -> 7.5, Float64[])
        @test LogDensityProblems.capabilities(p_empty) ==
            LogDensityProblems.LogDensityOrder{1}()
        val, grad = LogDensityProblems.logdensity_and_gradient(p_empty, Float64[])
        @test val == 7.5
        @test grad == Float64[]
    end
end
