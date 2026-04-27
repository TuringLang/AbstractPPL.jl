using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using LogDensityProblems: LogDensityProblems
using Test

struct _NTPrepared <: AbstractPPL.ADProblems.AbstractPrepared
    evaluator::AbstractPPL.ADProblems.NamedTupleEvaluator
end

@testset "AbstractPPLLogDensityProblemsExt" begin
    @testset "VectorEvaluator" begin
        ve = AbstractPPL.ADProblems.VectorEvaluator(sum, 3)
        @test LogDensityProblems.dimension(ve) == 3
        @test LogDensityProblems.logdensity(ve, [1.0, 2.0, 3.0]) == 6.0

        # trivial (dim=0) VectorEvaluator is gradient-capable
        ve0 = AbstractPPL.ADProblems.VectorEvaluator((_) -> 5.0, 0)
        @test LogDensityProblems.dimension(ve0) == 0
        @test LogDensityProblems.capabilities(ve0) ==
            LogDensityProblems.LogDensityOrder{1}()
        val0, grad0 = LogDensityProblems.logdensity_and_gradient(ve0, Float64[])
        @test val0 == 5.0
        @test grad0 == Float64[]
    end

    @testset "type-level capabilities" begin
        # Type-level dispatch follows the LDP convention (capabilities(ℓ) = capabilities(typeof(ℓ)))
        @test LogDensityProblems.capabilities(AbstractPPL.ADProblems.AbstractPrepared) ==
            LogDensityProblems.LogDensityOrder{0}()
    end

    @testset "NT-backed AbstractPrepared" begin
        # capabilities must be LogDensityOrder{0} for NT-backed prepared objects because
        # logdensity_and_gradient expects a NamedTuple, not the flat vector LDP callers pass.
        p = _NTPrepared(
            AbstractPPL.ADProblems.NamedTupleEvaluator(
                x -> x.a + sum(x.b), (a=0.0, b=zeros(2))
            ),
        )
        @test LogDensityProblems.capabilities(p) == LogDensityProblems.LogDensityOrder{0}()
    end
end
