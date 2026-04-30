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

# Backend extensions opt into gradient capability by overloading `capabilities`
# (typically on their cache type, e.g. `<:Prepared{<:Any,<:VectorEvaluator,<:MyCache}`).
# Here we dispatch on the AD type for simplicity.
function LogDensityProblems.capabilities(::Type{<:Prepared{TestADType,<:VectorEvaluator}})
    return LogDensityProblems.LogDensityOrder{1}()
end

@testset "AbstractPPLLogDensityProblemsExt" begin
    @testset "VectorEvaluator" begin
        ve = VectorEvaluator(sum, 3)
        @test LogDensityProblems.dimension(ve) == 3
        @test LogDensityProblems.logdensity(ve, [1.0, 2.0, 3.0]) == 6.0
        # A bare VectorEvaluator never advertises gradient capability;
        # only the wrapping `Prepared` does.
        @test LogDensityProblems.capabilities(ve) == LogDensityProblems.LogDensityOrder{0}()
    end

    @testset "Prepared capabilities" begin
        # Without a backend overload the fallback advertises order 0 only.
        p_no_overload = Prepared(AutoForwardDiff(), VectorEvaluator(sum, 3))
        @test LogDensityProblems.capabilities(p_no_overload) ==
            LogDensityProblems.LogDensityOrder{0}()

        # A backend that overloads capabilities advertises order 1.
        p_overloaded = Prepared(TestADType(), VectorEvaluator(sum, 3))
        @test LogDensityProblems.capabilities(p_overloaded) ==
            LogDensityProblems.LogDensityOrder{1}()
        @test LogDensityProblems.capabilities(typeof(p_overloaded)) ==
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
