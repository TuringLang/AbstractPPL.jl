using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using AbstractPPL.Evaluators: Prepared, VectorEvaluator, NamedTupleEvaluator
using ADTypes: AbstractADType, AutoForwardDiff
using LogDensityProblems: LogDensityProblems
using Test

# A NamedTupleEvaluator does not satisfy LDP's vector-input contract, so the
# extension does not define LDP methods for it.

struct TestADType <: AbstractADType end

function AbstractPPL.value_and_gradient!!(
    p::Prepared{TestADType}, x::AbstractVector{<:Real}
)
    return (p(x), ones(length(x)))
end

@testset "AbstractPPLLogDensityProblemsExt" begin
    @testset "VectorEvaluator" begin
        ve = VectorEvaluator(sum, 3)
        @test LogDensityProblems.dimension(ve) == 3
        @test LogDensityProblems.logdensity(ve, [1.0, 2.0, 3.0]) == 6.0
        # A bare evaluator (no `Prepared` wrapper) is primal-only.
        @test LogDensityProblems.capabilities(ve) == LogDensityProblems.LogDensityOrder{0}()
    end

    @testset "Prepared capabilities" begin
        # Any `Prepared` advertises order 1 — backends that don't implement
        # `value_and_gradient!!` will fail at call time, not via capabilities.
        p = Prepared(AutoForwardDiff(), VectorEvaluator(sum, 3))
        @test LogDensityProblems.capabilities(p) == LogDensityProblems.LogDensityOrder{1}()
        @test LogDensityProblems.capabilities(typeof(p)) ==
            LogDensityProblems.LogDensityOrder{1}()

        # NamedTupleEvaluator-backed Prepared has no LDP methods defined; the
        # extension only integrates vector-input evaluators.
        p_nt = Prepared(
            AutoForwardDiff(), NamedTupleEvaluator(x -> x.a + sum(x.b), (a=0.0, b=zeros(2)))
        )
        @test_throws MethodError LogDensityProblems.dimension(p_nt)
        @test LogDensityProblems.capabilities(p_nt) === nothing
    end

    @testset "logdensity_and_gradient" begin
        f = x -> -0.5 * sum(abs2, x)
        p = Prepared(TestADType(), VectorEvaluator(f, 3))
        x = [1.0, 2.0, 3.0]
        val, grad = LogDensityProblems.logdensity_and_gradient(p, x)
        @test val ≈ f(x)
        @test grad ≈ ones(3)
    end
end
