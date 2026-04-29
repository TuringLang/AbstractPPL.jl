using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using LogDensityProblems: LogDensityProblems
using Test

struct _VectorPrepared <: AbstractPPL.ADProblems.AbstractPrepared
    evaluator::AbstractPPL.ADProblems.VectorEvaluator
end

struct _NTPrepared <: AbstractPPL.ADProblems.AbstractPrepared
    evaluator::AbstractPPL.ADProblems.NamedTupleEvaluator
end

@testset "AbstractPPLLogDensityProblemsExt" begin
    @testset "VectorEvaluator" begin
        ve = AbstractPPL.ADProblems.VectorEvaluator(sum, 3)
        @test LogDensityProblems.dimension(ve) == 3
        @test LogDensityProblems.logdensity(ve, [1.0, 2.0, 3.0]) == 6.0
        # A bare VectorEvaluator never advertises gradient capability;
        # only the wrapping `AbstractPrepared` does.
        @test LogDensityProblems.capabilities(ve) == LogDensityProblems.LogDensityOrder{0}()
    end

    @testset "AbstractPrepared advertises gradient" begin
        p_vec = _VectorPrepared(AbstractPPL.ADProblems.VectorEvaluator(sum, 3))
        @test LogDensityProblems.capabilities(p_vec) ==
            LogDensityProblems.LogDensityOrder{1}()
        @test LogDensityProblems.capabilities(typeof(p_vec)) ==
            LogDensityProblems.LogDensityOrder{1}()

        p_nt = _NTPrepared(
            AbstractPPL.ADProblems.NamedTupleEvaluator(
                x -> x.a + sum(x.b), (a=0.0, b=zeros(2))
            ),
        )
        @test LogDensityProblems.capabilities(p_nt) ==
            LogDensityProblems.LogDensityOrder{1}()
    end
end
