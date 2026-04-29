using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using ForwardDiff
using Test

include(joinpath(@__DIR__, "..", "ad_tests.jl"))

@testset "AbstractPPLForwardDiffExt" begin
    x0 = zeros(3)
    x = [3.0, 1.0, 2.0]

    run_shared_gradient_tests(ADTypes.AutoForwardDiff(), x0, x)
    run_shared_jacobian_tests(ADTypes.AutoForwardDiff(), x0, [2.0, 3.0, 4.0])

    @testset "NamedTuple input" begin
        prepared = AbstractPPL.prepare(
            ADTypes.AutoForwardDiff(), vs -> vs.x^2 + sum(abs2, vs.y), (x=0.0, y=zeros(2))
        )
        @test prepared.evaluator isa AbstractPPL.ADProblems.NamedTupleEvaluator

        val, grad = AbstractPPL.value_and_gradient(prepared, (x=3.0, y=[1.0, 2.0]))
        @test val ≈ 14.0
        @test grad.x ≈ 6.0
        @test grad.y ≈ [2.0, 4.0]

        @test_throws r"same NamedTuple structure" AbstractPPL.value_and_gradient(
            prepared, (x=3.0, z=[1.0, 2.0])
        )
    end
end
