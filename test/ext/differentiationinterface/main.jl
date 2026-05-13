using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL: AbstractPPL, run_testcases
using ADTypes: AutoForwardDiff, AutoReverseDiff
using DifferentiationInterface
using ForwardDiff
using ReverseDiff
using Test

@testset "AbstractPPLDifferentiationInterfaceExt" begin
    run_testcases(Val(:vector); adtype=AutoForwardDiff(), atol=1e-6, rtol=1e-6)
    run_testcases(Val(:edge); adtype=AutoForwardDiff())

    @testset "AutoReverseDiff compiled tape (no-context path)" begin
        ad = AutoReverseDiff(; compile=true)
        p_scalar = AbstractPPL.prepare(ad, x -> sum(abs2, x), zeros(3))
        p_vector = AbstractPPL.prepare(ad, x -> [x[1] * x[2], x[2] + x[3]], zeros(3))

        @test !p_scalar.cache.use_context
        @test !isnothing(p_scalar.cache.gradient_prep.tape)
        @test !p_vector.cache.use_context
        @test !isnothing(p_vector.cache.jacobian_prep.tape)

        run_testcases(Val(:vector); adtype=ad, atol=1e-6, rtol=1e-6)
    end
end
