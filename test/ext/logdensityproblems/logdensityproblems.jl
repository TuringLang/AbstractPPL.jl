using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using AbstractPPL.ADProblems: Prepared, VectorEvaluator, NamedTupleEvaluator
using ADTypes: AutoForwardDiff
using LogDensityProblems: LogDensityProblems
using Test

@testset "AbstractPPLLogDensityProblemsExt" begin
    @testset "VectorEvaluator" begin
        ve = VectorEvaluator(sum, 3)
        @test LogDensityProblems.dimension(ve) == 3
        @test LogDensityProblems.logdensity(ve, [1.0, 2.0, 3.0]) == 6.0
        # A bare VectorEvaluator never advertises gradient capability;
        # only the wrapping `Prepared` does.
        @test LogDensityProblems.capabilities(ve) == LogDensityProblems.LogDensityOrder{0}()
    end

    @testset "Prepared advertises gradient" begin
        p_vec = Prepared(AutoForwardDiff(), VectorEvaluator(sum, 3))
        @test LogDensityProblems.capabilities(p_vec) ==
            LogDensityProblems.LogDensityOrder{1}()
        @test LogDensityProblems.capabilities(typeof(p_vec)) ==
            LogDensityProblems.LogDensityOrder{1}()

        p_nt = Prepared(
            AutoForwardDiff(),
            NamedTupleEvaluator(x -> x.a + sum(x.b), (a=0.0, b=zeros(2))),
        )
        @test LogDensityProblems.capabilities(p_nt) ==
            LogDensityProblems.LogDensityOrder{1}()
    end
end
