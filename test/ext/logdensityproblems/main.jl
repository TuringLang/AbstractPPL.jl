using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using AbstractPPL.Evaluators: Prepared, VectorEvaluator, NamedTupleEvaluator
using ADTypes: AbstractADType, AutoForwardDiff
using DifferentiationInterface: DifferentiationInterface as DI
using ForwardDiff
using LogDensityProblems: LogDensityProblems
using Test

@testset "AbstractPPLLogDensityProblemsExt" begin
    @testset "VectorEvaluator (no AD)" begin
        ve = VectorEvaluator(sum, 3)
        @test LogDensityProblems.dimension(ve) == 3
        @test LogDensityProblems.logdensity(ve, [1.0, 2.0, 3.0]) == 6.0
        @test LogDensityProblems.capabilities(ve) == LogDensityProblems.LogDensityOrder{0}()
    end

    @testset "Prepared without cache (no AD-aware prep)" begin
        p = Prepared(AutoForwardDiff(), VectorEvaluator(sum, 3))
        @test LogDensityProblems.dimension(p) == 3
        @test LogDensityProblems.logdensity(p, [1.0, 2.0, 3.0]) == 6.0
        @test LogDensityProblems.capabilities(p) == LogDensityProblems.LogDensityOrder{0}()
    end

    @testset "NamedTupleEvaluator-backed Prepared has no LDP methods" begin
        p_nt = Prepared(
            AutoForwardDiff(), NamedTupleEvaluator(x -> x.a + sum(x.b), (a=0.0, b=zeros(2)))
        )
        @test_throws MethodError LogDensityProblems.dimension(p_nt)
        @test LogDensityProblems.capabilities(p_nt) === nothing
    end

    @testset "DI cache shape drives capability" begin
        p_scalar = AbstractPPL.prepare(
            AutoForwardDiff(), x -> -0.5 * sum(abs2, x), zeros(3)
        )
        @test LogDensityProblems.capabilities(p_scalar) ==
            LogDensityProblems.LogDensityOrder{1}()
        x = [1.0, 2.0, 3.0]
        val, grad = LogDensityProblems.logdensity_and_gradient(p_scalar, x)
        @test val ≈ -0.5 * sum(abs2, x)
        @test grad ≈ -x

        p_vector = AbstractPPL.prepare(
            AutoForwardDiff(), x -> [x[1] * x[2], x[2] + x[3]], zeros(3)
        )
        @test LogDensityProblems.capabilities(p_vector) ==
            LogDensityProblems.LogDensityOrder{0}()

        p_empty_scalar = AbstractPPL.prepare(AutoForwardDiff(), x -> 7.5, Float64[])
        @test LogDensityProblems.capabilities(p_empty_scalar) ==
            LogDensityProblems.LogDensityOrder{1}()
        val, grad = LogDensityProblems.logdensity_and_gradient(p_empty_scalar, Float64[])
        @test val == 7.5
        @test grad == Float64[]

        p_empty_vector = AbstractPPL.prepare(AutoForwardDiff(), x -> [2.0, 3.0], Float64[])
        @test LogDensityProblems.capabilities(p_empty_vector) ==
            LogDensityProblems.LogDensityOrder{0}()
    end
end
