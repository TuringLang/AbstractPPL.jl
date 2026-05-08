using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL: run_testcases
using ADTypes: AutoForwardDiff, AutoReverseDiff
using DifferentiationInterface: DifferentiationInterface as DI
using ForwardDiff
using ReverseDiff
using Test

@testset "AbstractPPLDifferentiationInterfaceExt" begin
    @testset "ForwardDiff" begin
        run_testcases(Val(:vector); adtype=AutoForwardDiff(), atol=1e-6, rtol=1e-6)
        run_testcases(Val(:cache_reuse); adtype=AutoForwardDiff(), atol=1e-6, rtol=1e-6)
        run_testcases(Val(:edge); adtype=AutoForwardDiff())
    end

    # Compiled-tape ReverseDiff goes through the `_prepare_di(::AutoReverseDiff{true}, …)`
    # specialisation that closes the evaluator into a `Base.Fix2` target — the
    # `:cache_reuse` group exercises that path across multiple inputs.
    @testset "ReverseDiff (compiled tape)" begin
        adtype = AutoReverseDiff(; compile=true)
        run_testcases(Val(:vector); adtype=adtype, atol=1e-6, rtol=1e-6)
        run_testcases(Val(:cache_reuse); adtype=adtype, atol=1e-6, rtol=1e-6)
        run_testcases(Val(:edge); adtype=adtype)
    end
end
